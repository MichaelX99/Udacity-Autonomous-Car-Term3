#include "spline.h"
#include "behavior.h"
#include "helper.h"
#include <iostream>

Planner::Planner(Planner_Init input)
{
  _map_waypoints_x = input.map_waypoints_x;
  _map_waypoints_y = input.map_waypoints_y;
  _map_waypoints_s = input.map_waypoints_s;
  _map_waypoints_dx = input.map_waypoints_dx;
  _map_waypoints_dy = input.map_waypoints_dy;

  _lane = CENTER_LANE;
  _ref_vel = 0.0;
  _too_close = false;
}

std::vector<double> Planner::get_x_vals()
{
  return _next_x_vals;
}

std::vector<double> Planner::get_y_vals()
{
  return _next_y_vals;
}

int Planner::lane_keep()
{

  if (_prev_size > 0)
  {
    _car_s = _end_path_s;
  }

  _too_close = false;
  int car_ind = -1;

  for (int i = 0; i < _sensor_fusion.size(); i++)
  {
    float d = _sensor_fusion[i][6];
    if (d < (2 + 4 * _lane + 2) && d > (2 + 4 * _lane - 2))
    {
      double vx = _sensor_fusion[i][3];
      double vy = _sensor_fusion[i][4];
      double check_speed = sqrt(pow(vx,2) + pow(vy,2));
      double check_car_s = _sensor_fusion[i][5];

      check_car_s += ((double)(_prev_size) * 0.02 * check_speed);

      // if the car ahead is too close
      if ((check_car_s > _car_s) && ((check_car_s - _car_s) < 30))
      {
        _too_close = true;
        car_ind = i;
      }
    }
  }

  // Accelerate until we are at the speed limit if there is nothing preventing us
  if (_ref_vel < SPEED_LIMIT && !_too_close)
  {
    _ref_vel += SPEED_CHANGE;
  }

  return car_ind;
}

bool Planner::is_lane_clear(int lane)
{
  bool lane_clear = true;

  double next_to_thresh = 5;
  double in_front_thresh = 45;

  for (int i = 0; i < _sensor_fusion.size(); i++)
  {
    float d_check = _sensor_fusion[i][6];
    bool in_care_lane = (d_check < (2 + 4 * lane + 2)) && (d_check > (2 + 4 * lane - 2));

    if (in_care_lane)
    {
      double check_speed = sqrt(pow(_sensor_fusion[i][3],2) + pow(_sensor_fusion[i][4],2));
      double check_car_s = _sensor_fusion[i][5];
      check_car_s += ((double)(_prev_size) * 0.02 * check_speed);

      bool is_in_front = (check_car_s > _car_s) && ((check_car_s - _car_s) < in_front_thresh);
      bool is_next_to = abs(check_car_s - _car_s) < next_to_thresh;

      if (is_in_front || is_next_to)
      {
        lane_clear = false;
      }
    }
  }

  return lane_clear;
}


int Planner::lane_prepare()
{
  // If none of the other lanes are clear return the current lane
  int output = _lane;

  std::vector<int> lanes_to_check;
  // If we are not in the center lane then check to see if we can move there
  if (_lane == LEFT_LANE)
  {
    lanes_to_check.push_back(CENTER_LANE);
  }
  // If we are not in the left or right lane then check to see if we can move to the left lane
  else if (_lane == CENTER_LANE)
  {
    lanes_to_check.push_back(RIGHT_LANE);
    lanes_to_check.push_back(LEFT_LANE);
  }
  //  If we are not in the left or right lane check to see if we can move to the right lane
  else if (_lane == RIGHT_LANE)
  {
    lanes_to_check.push_back(CENTER_LANE);
  }

  for (size_t i = 0; i < lanes_to_check.size(); i++)
  {
    bool lane_clear = is_lane_clear(lanes_to_check[i]);
    if (lane_clear)
    {
      output = lanes_to_check[i];
    }
  }

  return output;
}

void Planner::lane_switch(int lane_to_switch)
{
  _lane = lane_to_switch;
}

void Planner::trajectory_generation()
{
  // Interpolated points to smooth path
  std::vector<double> ptsx;
  std::vector<double> ptsy;

  double ref_x = _car_x;
  double ref_y = _car_y;
  double ref_yaw = deg2rad(_car_yaw);

  // If we don't have enough points in our path then make some
  if (_prev_size < 2)
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
    ref_x = _previous_path_x[_prev_size-1];
    ref_y = _previous_path_y[_prev_size-1];

    double ref_x_prev = _previous_path_x[_prev_size-2];
    double ref_y_prev = _previous_path_y[_prev_size-2];
    ref_yaw = atan2(ref_y-ref_y_prev, ref_x-ref_x_prev);

    ptsx.push_back(ref_x_prev);
    ptsx.push_back(ref_x);

    ptsy.push_back(ref_y_prev);
    ptsy.push_back(ref_y);
  }

  // Get the XY coordinated of our path in the future
  std::vector<double> next_wp0 = getXY(_car_s+30, (2+4*_lane), _map_waypoints_s, _map_waypoints_x, _map_waypoints_y);
  std::vector<double> next_wp1 = getXY(_car_s+60, (2+4*_lane), _map_waypoints_s, _map_waypoints_x, _map_waypoints_y);
  std::vector<double> next_wp2 = getXY(_car_s+90, (2+4*_lane), _map_waypoints_s, _map_waypoints_x, _map_waypoints_y);

  ptsx.push_back(next_wp0[0]);
  ptsx.push_back(next_wp1[0]);
  ptsx.push_back(next_wp2[0]);

  ptsy.push_back(next_wp0[1]);
  ptsy.push_back(next_wp1[1]);
  ptsy.push_back(next_wp2[1]);

  // Transform our coordinate system to make it look like our vehicle is facing the direction of the point not from its current orientation
  for (int i = 0; i < ptsx.size(); i++)
  {
    double shift_x = ptsx[i] - ref_x;
    double shift_y = ptsy[i] - ref_y;

    ptsx[i] = (shift_x * cos(0-ref_yaw) - shift_y * sin(0-ref_yaw));
    ptsy[i] = (shift_x * sin(0-ref_yaw) + shift_y * cos(0-ref_yaw));
  }

  // Define the spline object
  tk::spline s;

  // Fill the spline with the path that we  generated into the future
  s.set_points(ptsx, ptsy);

  // If there are any points from the earlier iteration make sure that those also get executed
  for (int i = 0; i < _prev_size; i++)
  {
    _next_x_vals.push_back(_previous_path_x[i]);
    _next_y_vals.push_back(_previous_path_y[i]);
  }


  double target_x = 30.0;
  double target_y = s(target_x);
  double target_dist = sqrt(pow(target_x,2) + pow(target_y,2));

  double x_add_on = 0;

  // For all the points in our path that we still need to populate
  for (int i = 1; i <= PATH_SIZE - _prev_size; i++)
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

  _prev_size = _previous_path_x.size();

  // Keep our current lane and check if we need to leave the lane
  int car_ind = lane_keep();

  // Is there a vehicle blocking our way?
  if (_too_close)
  {
    // Determine whether it makes sense to change lanes
    int should_switch = lane_prepare();

    // If the best lane is not the one we are currently in then change lanes
    if (should_switch != _lane)
    {
      lane_switch(should_switch);
    }
    // If there is no good lane to change to then just slow down
    else
    {
      double vx = _sensor_fusion[car_ind][3];
      double vy = _sensor_fusion[car_ind][4];
      double check_speed = sqrt(pow(vx,2) + pow(vy,2));

      double diff = _ref_vel - check_speed;
      _ref_vel -= diff * .009;
      //_ref_vel -= SPEED_CHANGE;
    }
  }
  // If there is no vehicle in our way and we are not in the center lane
  else
  {
    if (_lane != CENTER_LANE)
    {
      // If the center lane is clear switch to it
      if (is_lane_clear(CENTER_LANE))
      {
        lane_switch(CENTER_LANE);
      }
    }
  }

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
