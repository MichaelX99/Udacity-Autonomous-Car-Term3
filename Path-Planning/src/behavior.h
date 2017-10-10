#ifndef BEHAVIOR
#define BEHAVIOR

#include <vector>

#define PATH_SIZE 50

#define SPEED_CHANGE .224

#define SPEED_LIMIT 49.5

#define LEFT_LANE 0
#define CENTER_LANE 1
#define RIGHT_LANE 2

struct Sim_Input
{
  double car_x;
  double car_y;
  double car_s;
  double car_d;
  double car_yaw;
  double car_speed;

  std::vector<double> previous_path_x;
  std::vector<double> previous_path_y;

  double end_path_s;
  double end_path_d;

  std::vector<std::vector<double> > sensor_fusion;
};

struct Planner_Init
{
  std::vector<double> map_waypoints_x;
  std::vector<double> map_waypoints_y;
  std::vector<double> map_waypoints_s;
  std::vector<double> map_waypoints_dx;
  std::vector<double> map_waypoints_dy;
};

class Planner
{
public:
  Planner(Planner_Init input);


  void plan(Sim_Input input);
  std::vector<double> get_x_vals();
  std::vector<double> get_y_vals();

  void clear();

private:
  // Values to populate to output to the simulator
  std::vector<double> _next_x_vals;
  std::vector<double> _next_y_vals;

  // Define the map
  std::vector<double> _map_waypoints_x;
  std::vector<double> _map_waypoints_y;
  std::vector<double> _map_waypoints_s;
  std::vector<double> _map_waypoints_dx;
  std::vector<double> _map_waypoints_dy;

  // current lane
  int _lane;

  // reference velocity
  double _ref_vel;

  bool _too_close;

  int _prev_size;

  // Inputs from the simulator at each time step
  double _car_x;
  double _car_y;
  double _car_s;
  double _car_d;
  double _car_yaw;
  double _car_speed;
  std::vector<double> _previous_path_x;
  std::vector<double> _previous_path_y;
  double _end_path_s;
  double _end_path_d;
  std::vector<std::vector<double> > _sensor_fusion;


  // Class functions
  int lane_keep();
  int lane_prepare();
  bool is_lane_clear(int lane);
  void lane_switch(int lane_to_switch);
  void trajectory_generation();

};


#endif
