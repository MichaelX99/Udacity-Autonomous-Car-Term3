#include "spline.h"
#include "behavior.h"
#include "helper.h"

Planner::Planner(Planner_Init input)
{
  _map_waypoints_x = input.map_waypoints_x;
  _map_waypoints_y = input.map_waypoints_y;
  _map_waypoints_s = input.map_waypoints_s;
  _map_waypoints_dx = input.map_waypoints_dx;
  _map_waypoints_dy = input.map_waypoints_dy;

  _lane = 1;
  _ref_vel = 0.0;
}

std::vector<double> Planner::get_x_vals()
{
  return _next_x_vals;
}

std::vector<double> Planner::get_y_vals()
{
  return _next_y_vals;
}

void Planner::avoid_vehicles()
{
  int prev_size = _previous_path_x.size();

  if (prev_size > 0)
  {
    _car_s = _end_path_s;
  }

  bool too_close = false;

  for (int i = 0; i < _sensor_fusion.size(); i++)
  {
    float d = _sensor_fusion[i][6];
    if (d < (2 + 4 * _lane + 2) && d > (2 + 4 * _lane - 2))
    {
      double vx = _sensor_fusion[i][3];
      double vy = _sensor_fusion[i][4];
      double check_speed = sqrt(pow(vx,2) + pow(vy,2));
      double check_car_s = _sensor_fusion[i][5];

      check_car_s += ((double)(prev_size) * 0.02 * check_speed);

      if ((check_car_s > _car_s) && ((check_car_s - _car_s) < 30))
      {
        too_close = true;
        // SHOWS HOW TO CHANGE LANES
        ////////////////////////////////////////////////////////////////////////////////////////////////////
        if (_lane > 0)
        {
          _lane = 0;
        }
      }
    }
  }

  // SHOWS HOW TO SLOW DOWN NOT STOP
  //////////////////////////////////////////////////////////////////////////////////////////////////////////
  if (too_close)
  {
    _ref_vel -= .224;
  }
  else if (_ref_vel < SPEED_LIMIT)
  {
    _ref_vel += .224;
  }
}

void Planner::trajectory_generation()
{
  int prev_size = _previous_path_x.size();

  std::vector<double> ptsx;
  std::vector<double> ptsy;

  double ref_x = _car_x;
  double ref_y = _car_y;
  double ref_yaw = deg2rad(_car_yaw);

  if (prev_size < 2)
  {
    double prev_car_x = _car_x - cos(_car_yaw);
    double prev_car_y = _car_y - sin(_car_yaw);

    ptsx.push_back(prev_car_x);
    ptsx.push_back(_car_x);

    ptsy.push_back(prev_car_y);
    ptsy.push_back(_car_y);
  }
  else
  {
    ref_x = _previous_path_x[prev_size-1];
    ref_y = _previous_path_y[prev_size-1];

    double ref_x_prev = _previous_path_x[prev_size-2];
    double ref_y_prev = _previous_path_y[prev_size-2];
    ref_yaw = atan2(ref_y-ref_y_prev, ref_x-ref_x_prev);

    ptsx.push_back(ref_x_prev);
    ptsx.push_back(ref_x);

    ptsy.push_back(ref_y_prev);
    ptsy.push_back(ref_y);
  }

  std::vector<double> next_wp0 = getXY(_car_s+30, (2+4*_lane), _map_waypoints_s, _map_waypoints_x, _map_waypoints_y);
  std::vector<double> next_wp1 = getXY(_car_s+60, (2+4*_lane), _map_waypoints_s, _map_waypoints_x, _map_waypoints_y);
  std::vector<double> next_wp2 = getXY(_car_s+90, (2+4*_lane), _map_waypoints_s, _map_waypoints_x, _map_waypoints_y);

  ptsx.push_back(next_wp0[0]);
  ptsx.push_back(next_wp1[0]);
  ptsx.push_back(next_wp2[0]);

  ptsy.push_back(next_wp0[1]);
  ptsy.push_back(next_wp1[1]);
  ptsy.push_back(next_wp2[1]);

  for (int i = 0; i < ptsx.size(); i++)
  {
    double shift_x = ptsx[i] - ref_x;
    double shift_y = ptsy[i] - ref_y;

    ptsx[i] = (shift_x * cos(0-ref_yaw) - shift_y * sin(0-ref_yaw));
    ptsy[i] = (shift_x * sin(0-ref_yaw) + shift_y * cos(0-ref_yaw));
  }

  tk::spline s;

  s.set_points(ptsx, ptsy);

  for (int i = 0; i < prev_size; i++)
  {
    _next_x_vals.push_back(_previous_path_x[i]);
    _next_y_vals.push_back(_previous_path_y[i]);
  }

  double target_x = 30.0;
  double target_y = s(target_x);
  double target_dist = sqrt(pow(target_x,2) + pow(target_y,2));

  double x_add_on = 0;

  for (int i = 1; i <= 50 - prev_size; i++)
  {
    double N = target_dist / (0.02 * _ref_vel / 2.24);
    double x_point = x_add_on + target_x/N;
    double y_point = s(x_point);

    x_add_on = x_point;

    double x_ref = x_point;
    double y_ref = y_point;

    x_point = (x_ref * cos(ref_yaw) - y_ref * sin(ref_yaw));
    y_point = (x_ref * sin(ref_yaw) + y_ref * cos(ref_yaw));

    x_point += ref_x;
    y_point += ref_y;

    _next_x_vals.push_back(x_point);
    _next_y_vals.push_back(y_point);
  }

}

void Planner::plan(Sim_Input input)
{
  _car_x = input.car_x;
  _car_y = input.car_y;
  _car_s = input.car_s;
  _car_d = input.car_d;
  _car_yaw = input.car_yaw;
  _car_speed = input.car_speed;
  _previous_path_x = input.previous_path_x;
  _previous_path_y = input.previous_path_y;
  _end_path_s = input.end_path_s;
  _end_path_d = input.end_path_d;
  _sensor_fusion = input.sensor_fusion;

  avoid_vehicles();
  trajectory_generation();
}

void Planner::clear()
{
  _previous_path_x.clear();
  _previous_path_y.clear();
  for (int i = 0; i < _sensor_fusion.size(); i++)
  {
    _sensor_fusion[i].clear();
  }
  _sensor_fusion.clear();
  _next_x_vals.clear();
  _next_y_vals.clear();
}
