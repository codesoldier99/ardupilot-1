#include <AP_HAL/AP_HAL.h>
#include "AP_L1_Control.h"
#include <GCS_MAVLink/GCS.h>
#include <AP_Logger/AP_logger.h>
#include <math.h>
#include "unit.h"
#include "DDPG_C.h"



int32_t num=1;
int32_t  maxradius=1;

float k1 = 1.0f;
int Lstate = 0;
float Laction = 4.0f;
int Lstate3d = 0;
float k3 = 1.0f;






extern const AP_HAL::HAL& hal;

// table of user settable parameters
const AP_Param::GroupInfo AP_L1_Control::var_info[] = {
    // @Param: PERIOD
    // @DisplayName: L1 control period
    // @Description: Period in seconds of L1 tracking loop. This parameter is the primary control for agressiveness of turns in auto mode. This needs to be larger for less responsive airframes. The default of 20 is quite conservative, but for most RC aircraft will lead to reasonable flight. For smaller more agile aircraft a value closer to 15 is appropriate, or even as low as 10 for some very agile aircraft. When tuning, change this value in small increments, as a value that is much too small (say 5 or 10 below the right value) can lead to very radical turns, and a risk of stalling.
    // @Units: s
    // @Range: 1 60
    // @Increment: 1
    // @User: Standard
    AP_GROUPINFO("PERIOD",    0, AP_L1_Control, _L1_period, 17),

    // @Param: DAMPING
    // @DisplayName: L1 control damping ratio
    // @Description: Damping ratio for L1 control. Increase this in increments of 0.05 if you are getting overshoot in path tracking. You should not need a value below 0.7 or above 0.85.
    // @Range: 0.6 1.0
    // @Increment: 0.05
    // @User: Advanced
    AP_GROUPINFO("DAMPING",   1, AP_L1_Control, _L1_damping, 0.75f),

    // @Param: XTRACK_I
    // @DisplayName: L1 control crosstrack integrator gain
    // @Description: Crosstrack error integrator gain. This gain is applied to the crosstrack error to ensure it converges to zero. Set to zero to disable. Smaller values converge slower, higher values will cause crosstrack error oscillation.
    // @Range: 0 0.1
    // @Increment: 0.01
    // @User: Advanced
    AP_GROUPINFO("XTRACK_I",   2, AP_L1_Control, _L1_xtrack_i_gain, 0.02),

    // @Param: LIM_BANK
    // @DisplayName: Loiter Radius Bank Angle Limit
    // @Description: The sealevel bank angle limit for a continous loiter. (Used to calculate airframe loading limits at higher altitudes). Setting to 0, will instead just scale the loiter radius directly
    // @Units: deg
    // @Range: 0 89
    // @User: Advanced
    AP_GROUPINFO_FRAME("LIM_BANK",   3, AP_L1_Control, _loiter_bank_limit, 0.0f, AP_PARAM_FRAME_PLANE),

    AP_GROUPEND
};

//Bank angle command based on angle between aircraft velocity vector and reference vector to path.
//S. Park, J. Deyst, and J. P. How, "A New Nonlinear Guidance Logic for Trajectory Tracking,"
//Proceedings of the AIAA Guidance, Navigation and Control
//Conference, Aug 2004. AIAA-2004-4900.
//Modified to use PD control for circle tracking to enable loiter radius less than L1 length
//Modified to enable period and damping of guidance loop to be set explicitly
//Modified to provide explicit control over capture angle



/*
  Wrap AHRS yaw if in reverse - radians
 */
float AP_L1_Control::get_yaw()
{
    if (_reverse) {
        return wrap_PI(M_PI + _ahrs.yaw);
    }
    return _ahrs.yaw;
}

/*
  Wrap AHRS yaw sensor if in reverse - centi-degress
 */
float AP_L1_Control::get_yaw_sensor()
{
    if (_reverse) {
        return wrap_180_cd(18000 + _ahrs.yaw_sensor);
    }
    return _ahrs.yaw_sensor;
}

/*
  return the bank angle needed to achieve tracking from the last
  update_*() operation
 */
int32_t AP_L1_Control::nav_roll_cd(void) const
{
    float ret;
    ret = cosf(_ahrs.pitch)*degrees(atanf(_latAccDem * 0.101972f) * 100.0f); // 0.101972 = 1/9.81
    ret = constrain_float(ret, -9000, 9000);
    return ret;
}

/*
  return the lateral acceleration needed to achieve tracking from the last
  update_*() operation
 */
float AP_L1_Control::lateral_acceleration(void) const
{
    return _latAccDem;
}

int32_t AP_L1_Control::nav_bearing_cd(void) const
{
    return wrap_180_cd(RadiansToCentiDegrees(_nav_bearing));
}

int32_t AP_L1_Control::bearing_error_cd(void) const
{
    return RadiansToCentiDegrees(_bearing_error);
}

int32_t AP_L1_Control::target_bearing_cd(void) const
{
    return wrap_180_cd(_target_bearing_cd);
}

/*
  this is the turn distance assuming a 90 degree turn
 */
float AP_L1_Control::turn_distance(float wp_radius) const
{
    wp_radius *= sq(_ahrs.get_EAS2TAS());
    return MIN(wp_radius, _L1_dist);
}

/*
  this approximates the turn distance for a given turn angle. If the
  turn_angle is > 90 then a 90 degree turn distance is used, otherwise
  the turn distance is reduced linearly.
  This function allows straight ahead mission legs to avoid thinking
  they have reached the waypoint early, which makes things like camera
  trigger and ball drop at exact positions under mission control much easier
 */
float AP_L1_Control::turn_distance(float wp_radius, float turn_angle) const
{
    float distance_90 = turn_distance(wp_radius);
    turn_angle = fabsf(turn_angle);
    if (turn_angle >= 90) {
        return distance_90;
    }
    return distance_90 * turn_angle / 90.0f;
}

float AP_L1_Control::loiter_radius(const float radius) const
{
    // prevent an insane loiter bank limit
    float sanitized_bank_limit = constrain_float(_loiter_bank_limit, 0.0f, 89.0f);
    float lateral_accel_sea_level = tanf(radians(sanitized_bank_limit)) * GRAVITY_MSS;

    float nominal_velocity_sea_level;
    if(_spdHgtControl == nullptr) {
        nominal_velocity_sea_level = 0.0f;
    } else {
        nominal_velocity_sea_level =  _spdHgtControl->get_target_airspeed();
    }

    float eas2tas_sq = sq(_ahrs.get_EAS2TAS());

    if (is_zero(sanitized_bank_limit) || is_zero(nominal_velocity_sea_level) ||
        is_zero(lateral_accel_sea_level)) {
        // Missing a sane input for calculating the limit, or the user has
        // requested a straight scaling with altitude. This will always vary
        // with the current altitude, but will at least protect the airframe
        return radius * eas2tas_sq;
    } else {
        float sea_level_radius = sq(nominal_velocity_sea_level) / lateral_accel_sea_level;
        if (sea_level_radius > radius) {
            // If we've told the plane that its sea level radius is unachievable fallback to
            // straight altitude scaling
            return radius * eas2tas_sq;
        } else {
            // select the requested radius, or the required altitude scale, whichever is safer
            return MAX(sea_level_radius * eas2tas_sq, radius);
        }
    }
}

bool AP_L1_Control::reached_loiter_target(void)
{
    return _WPcircle;
}

/**
   prevent indecision in our turning by using our previous turn
   decision if we are in a narrow angle band pointing away from the
   target and the turn angle has changed sign
 */
void AP_L1_Control::_prevent_indecision(float &Nu)
{
    const float Nu_limit = 0.9f*M_PI;
    if (fabsf(Nu) > Nu_limit &&
        fabsf(_last_Nu) > Nu_limit &&
        labs(wrap_180_cd(_target_bearing_cd - get_yaw_sensor())) > 12000 &&
        Nu * _last_Nu < 0.0f) {
        // we are moving away from the target waypoint and pointing
        // away from the waypoint (not flying backwards). The sign
        // of Nu has also changed, which means we are
        // oscillating in our decision about which way to go
        Nu = _last_Nu;
    }
}

// update L1 control for waypoint navigation
void AP_L1_Control::update_waypoint(const struct Location &prev_WP, const struct Location &next_WP, float dist_min)
{

    struct Location _current_loc;
    float Nu;
    float xtrackVel;
    float ltrackVel;

    uint32_t now = AP_HAL::micros();
    float dt = (now - _last_update_waypoint_us) * 1.0e-6f;
    if (dt > 0.1) {
        dt = 0.1;
        _L1_xtrack_i = 0.0f;
    }
    _last_update_waypoint_us = now;

	

    // Calculate L1 gain required for specified damping
    //float K_L1 = 4.0f * _L1_damping * _L1_damping;

    // Get current position and velocity
    if (_ahrs.get_position(_current_loc) == false) {
        // if no GPS loc available, maintain last nav/target_bearing
        _data_is_stale = true;
        return;
    }

    Vector2f _groundspeed_vector = _ahrs.groundspeed_vector();

    // update _target_bearing_cd
    _target_bearing_cd = _current_loc.get_bearing_to(next_WP);

    //Calculate groundspeed
    float groundSpeed = _groundspeed_vector.length();
    if (groundSpeed < 0.1f) {
        // use a small ground speed vector in the right direction,
        // allowing us to use the compass heading at zero GPS velocity
        groundSpeed = 0.1f;
        _groundspeed_vector = Vector2f(cosf(get_yaw()), sinf(get_yaw())) * groundSpeed;
    }

    // Calculate time varying control parameters
    // Calculate the L1 length required for specified period
    // 0.3183099 = 1/1/pipi
    _L1_dist = MAX(0.3183099f * _L1_damping * _L1_period * groundSpeed, dist_min);

    // Calculate the NE position of WP B relative to WP A
    Vector2f AB = prev_WP.get_distance_NE(next_WP);
    float AB_length = AB.length();

    // Check for AB zero length and track directly to the destination
    // if too small
    if (AB.length() < 1.0e-6f) {
        AB = _current_loc.get_distance_NE(next_WP);
        if (AB.length() < 1.0e-6f) {
            AB = Vector2f(cosf(get_yaw()), sinf(get_yaw()));
        }
    }
    AB.normalize();

    // Calculate the NE position of the aircraft relative to WP A
    const Vector2f A_air = prev_WP.get_distance_NE(_current_loc);

    // calculate distance to target track, for reporting
    _crosstrack_error = A_air % AB;
	//gcs().send_text(MAV_SEVERITY_INFO, "Tuning: cerr %.2f",(float)_crosstrack_error);

    //Determine if the aircraft is behind a +-135 degree degree arc centred on WP A
    //and further than L1 distance from WP A. Then use WP A as the L1 reference point
    //Otherwise do normal L1 guidance
    float WP_A_dist = A_air.length();
    float alongTrackDist = A_air * AB;

	v1=AB_length/2e+03;
	v2=WP_A_dist/2e+03;
	v3=(_crosstrack_error-(-80.0))/(50.0-(-80.0));;
	v4=groundSpeed/30.0;
	//gcs().send_text(MAV_SEVERITY_INFO, "Tuning: v1 %d",(int)v1);
	//Qleaning
	if (_crosstrack_error<0.1f)
		{
		Lstate = 0;
	}
	else if(_crosstrack_error<0.1f && _crosstrack_error >-0.1f)
		{
		Lstate = 1;
	}
	else {
     Lstate = 2;
	}
	double Q[100][100]= {{0,0,500},
	                 {0,500,400},
	                 {0,0,500}};
	int best_action = inference_best_action(Lstate, Q);
	if (best_action ==0)
		{
		k1 = 3.8f;
	}
	else if(best_action ==1)
		{
		k1 = 4.0f;

	}
	else if(best_action == 2)
		{
		k1 = 6.0f;
	}

	
   float K_L1 = 4.0f * _L1_damping * _L1_damping;

   AP::logger().Write("L1", "TimeUS,Err,l1,ab", "Qfff",
                                               AP_HAL::micros64(),
                                               (double)_crosstrack_error,
                                               (double)_L1_dist,
                                               (double)AB_length
                                               );
	
    if (WP_A_dist > _L1_dist && alongTrackDist/MAX(WP_A_dist, 1.0f) < -0.7071f)
    {
        //Calc Nu to fly To WP A
        Vector2f A_air_unit = (A_air).normalized(); // Unit vector from WP A to aircraft
        xtrackVel = _groundspeed_vector % (-A_air_unit); // Velocity across line
        ltrackVel = _groundspeed_vector * (-A_air_unit); // Velocity along line
        Nu = atan2f(xtrackVel,ltrackVel);
        _nav_bearing = atan2f(-A_air_unit.y , -A_air_unit.x); // bearing (radians) from AC to L1 point
    } else if (alongTrackDist > AB_length + groundSpeed*3) {
        // we have passed point B by 3 seconds. Head towards B
        // Calc Nu to fly To WP B
        const Vector2f B_air = next_WP.get_distance_NE(_current_loc);
        Vector2f B_air_unit = (B_air).normalized(); // Unit vector from WP B to aircraft
        xtrackVel = _groundspeed_vector % (-B_air_unit); // Velocity across line
        ltrackVel = _groundspeed_vector * (-B_air_unit); // Velocity along line
        Nu = atan2f(xtrackVel,ltrackVel);
        _nav_bearing = atan2f(-B_air_unit.y , -B_air_unit.x); // bearing (radians) from AC to L1 point
    } else { //Calc Nu to fly along AB line

        //Calculate Nu2 angle (angle of velocity vector relative to line connecting waypoints)
        xtrackVel = _groundspeed_vector % AB; // Velocity cross track
        ltrackVel = _groundspeed_vector * AB; // Velocity along track
        float Nu2 = atan2f(xtrackVel,ltrackVel);
        //Calculate Nu1 angle (Angle to L1 reference point)
        float sine_Nu1 = _crosstrack_error/MAX(_L1_dist, 0.1f);
        //Limit sine of Nu1 to provide a controlled track capture angle of 45 deg
        sine_Nu1 = constrain_float(sine_Nu1, -0.7071f, 0.7071f);
        float Nu1 = asinf(sine_Nu1);

        // compute integral error component to converge to a crosstrack of zero when traveling
        // straight but reset it when disabled or if it changes. That allows for much easier
        // tuning by having it re-converge each time it changes.
        if (_L1_xtrack_i_gain <= 0 || !is_equal(_L1_xtrack_i_gain.get(), _L1_xtrack_i_gain_prev)) {
            _L1_xtrack_i = 0;
            _L1_xtrack_i_gain_prev = _L1_xtrack_i_gain;
        } else if (fabsf(Nu1) < radians(5)) {
            _L1_xtrack_i += Nu1 * _L1_xtrack_i_gain * dt;

            // an AHRS_TRIM_X=0.1 will drift to about 0.08 so 0.1 is a good worst-case to clip at
            _L1_xtrack_i = constrain_float(_L1_xtrack_i, -0.1f, 0.1f);
        }

        // to converge to zero we must push Nu1 harder
        Nu1 += _L1_xtrack_i;

        Nu = Nu1 + Nu2;
        _nav_bearing = atan2f(AB.y, AB.x) + Nu1; // bearing (radians) from AC to L1 point
    }

    _prevent_indecision(Nu);
    _last_Nu = Nu;

    //Limit Nu to +-(pi/2)
    Nu = constrain_float(Nu, -1.5708f, +1.5708f);
    _latAccDem = K_L1 * groundSpeed * groundSpeed / _L1_dist * sinf(Nu);

    // Waypoint capture status is always false during waypoint following
    _WPcircle = false;

    _bearing_error = Nu; // bearing error angle (radians), +ve to left of track

    _data_is_stale = false; // status are correctly updated with current waypoint data
}

// update L1 control for loitering
void AP_L1_Control::update_loiter(const struct Location &center_WP, float radius, int8_t loiter_direction)
{
    struct Location _current_loc;

    // scale loiter radius with square of EAS2TAS to allow us to stay
    // stable at high altitude
    radius = loiter_radius(fabsf(radius));

    // Calculate guidance gains used by PD loop (used during circle tracking)
    float omega = (6.2832f / _L1_period);
    float Kx = omega * omega;
    float Kv = 2.0f * _L1_damping * omega;

    // Calculate L1 gain required for specified damping (used during waypoint capture)
    float K_L1 = 4.0f * _L1_damping * _L1_damping;

    //Get current position and velocity
    if (_ahrs.get_position(_current_loc) == false) {
        // if no GPS loc available, maintain last nav/target_bearing
        _data_is_stale = true;
        return;
    }

    Vector2f _groundspeed_vector = _ahrs.groundspeed_vector();

    //Calculate groundspeed
    float groundSpeed = MAX(_groundspeed_vector.length() , 1.0f);


    // update _target_bearing_cd
    _target_bearing_cd = _current_loc.get_bearing_to(center_WP);


    // Calculate time varying control parameters
    // Calculate the L1 length required for specified period
    // 0.3183099 = 1/pi
    _L1_dist = 0.3183099f * _L1_damping * _L1_period * groundSpeed;

    //Calculate the NE position of the aircraft relative to WP A
    const Vector2f A_air = center_WP.get_distance_NE(_current_loc);

    // Calculate the unit vector from WP A to aircraft
    // protect against being on the waypoint and having zero velocity
    // if too close to the waypoint, use the velocity vector
    // if the velocity vector is too small, use the heading vector
    Vector2f A_air_unit;
    if (A_air.length() > 0.1f) {
        A_air_unit = A_air.normalized();
    } else {
        if (_groundspeed_vector.length() < 0.1f) {
            A_air_unit = Vector2f(cosf(_ahrs.yaw), sinf(_ahrs.yaw));
        } else {
            A_air_unit = _groundspeed_vector.normalized();
        }
    }

    //Calculate Nu to capture center_WP
    float xtrackVelCap = A_air_unit % _groundspeed_vector; // Velocity across line - perpendicular to radial inbound to WP
    float ltrackVelCap = - (_groundspeed_vector * A_air_unit); // Velocity along line - radial inbound to WP
    float Nu = atan2f(xtrackVelCap,ltrackVelCap);

    _prevent_indecision(Nu);
    _last_Nu = Nu;

    Nu = constrain_float(Nu, -M_PI_2, M_PI_2); //Limit Nu to +- Pi/2

    //Calculate lat accln demand to capture center_WP (use L1 guidance law)
    float latAccDemCap = K_L1 * groundSpeed * groundSpeed / _L1_dist * sinf(Nu);

    //Calculate radial position and velocity errors
    float xtrackVelCirc = -ltrackVelCap; // Radial outbound velocity - reuse previous radial inbound velocity
    float xtrackErrCirc = A_air.length() - radius; // Radial distance from the loiter circle

    // keep crosstrack error for reporting
    _crosstrack_error = xtrackErrCirc;

    //Calculate PD control correction to circle waypoint_ahrs.roll
    float latAccDemCircPD = (xtrackErrCirc * Kx + xtrackVelCirc * Kv);

    //Calculate tangential velocity
    float velTangent = xtrackVelCap * float(loiter_direction);

    //Prevent PD demand from turning the wrong way by limiting the command when flying the wrong way
    if (ltrackVelCap < 0.0f && velTangent < 0.0f) {
        latAccDemCircPD =  MAX(latAccDemCircPD, 0.0f);
    }

    // Calculate centripetal acceleration demand
    float latAccDemCircCtr = velTangent * velTangent / MAX((0.5f * radius), (radius + xtrackErrCirc));

    //Sum PD control and centripetal acceleration to calculate lateral manoeuvre demand
    float latAccDemCirc = loiter_direction * (latAccDemCircPD + latAccDemCircCtr);

    // Perform switchover between 'capture' and 'circle' modes at the
    // point where the commands cross over to achieve a seamless transfer
    // Only fly 'capture' mode if outside the circle
    if (xtrackErrCirc > 0.0f && loiter_direction * latAccDemCap < loiter_direction * latAccDemCirc) {
        _latAccDem = latAccDemCap;
        _WPcircle = false;
        _bearing_error = Nu; // angle between demanded and achieved velocity vector, +ve to left of track
        _nav_bearing = atan2f(-A_air_unit.y , -A_air_unit.x); // bearing (radians) from AC to L1 point
    } else {
        _latAccDem = latAccDemCirc;
        _WPcircle = true;
        _bearing_error = 0.0f; // bearing error (radians), +ve to left of track
        _nav_bearing = atan2f(-A_air_unit.y , -A_air_unit.x); // bearing (radians)from AC to L1 point
    }

    _data_is_stale = false; // status are correctly updated with current waypoint data
}

//修改代码
void AP_L1_Control::update_loiter_ellipse(const struct Location &center_loc, const int32_t maxradius_cm, const float minmaxratio, const float psi, const int8_t orientation, struct Location &aircraft_loc, Vector3f &aircraft_vel, struct Location &desired_loc)
{
    struct Location _current_loc;
	//int32_t  maxradius;
	//int32_t  g;
	//int32_t fla1=0;
	//int32_t fla2=0;

    // scale loiter radius with square of EAS2TAS to allow us to stay
    // stable at high altitude
    // maxradius_cm *= sq(_ahrs.get_EAS2TAS());

    // Calculate guidance gains used by PD loop (used during circle tracking)
    const float omega = (6.2832f / _L1_period);
    const float Kx = omega * omega;
    const float Kv = 2.0f * _L1_damping * omega;

    // Calculate L1 gain required for specified damping (used during waypoint capture)
    //const float K_L1 = 4.0f * _L1_damping * _L1_damping;
    const float K_L1 = 8.0f * _L1_damping * _L1_damping;

    // get current position and velocity in NED frame
    if (_ahrs.get_position(_current_loc) == false) {
        // if no GPS loc available, maintain last nav/target_bearing
        _data_is_stale = true;
        return;
    }
        // position of aircraft relative to the center of the ellipse; vector components in meters
    const Vector3f posav(center_loc.location_3d_diff_NED(_current_loc));
    // store absolute aircraft position for external use
    aircraft_loc = _current_loc;

    // lateral projection
    const Vector2f posalv(posav.x, posav.y);
	//gcs().send_text(MAV_SEVERITY_INFO, "Tuning: x,y %.2f %.2f",posav.x,posav.y);

    // update _target_bearing_cd
    _target_bearing_cd = _current_loc.get_bearing_to(center_loc);
	gcs().send_text(MAV_SEVERITY_INFO, "Tuning: bearing %d",(int)_target_bearing_cd);
	
	   //每半圈改变无人机的半径 
	   /*maxradius= maxradius_cm;
	if ((_target_bearing_cd >= 18000 && _target_bearing_cd < 18100)||(_target_bearing_cd < 36000 && _target_bearing_cd > 35900))
		{
	    
		if ((_target_bearing_cd-18000)<40 || abs(_target_bearing_cd-36000)<40)
			{
		   		num=num+1;//n代表n个半圈
		   		gcs().send_text(MAV_SEVERITY_INFO, "Tuning: n %d",(int)num);
			}
		}*/
		
      /*if(fla1==0 && _target_bearing_cd<=18000)
      	{  fla2=0; 
      	  num=num+1;
		 fla1=1;
      	}
	 
      	if(fla2==0 && (_target_bearing_cd>18000 && _target_bearing_cd<=36000))
      	{  fla1=0;
      	  num=num+1;
		  fla2=1;
      	}*/
	  
      //maxradius=maxradius*pow(1.1, num);
	
	/*for (g = 0; g < num; ++g)
		{
		maxradius=maxradius*1.1;
		}*/
		


    // velocity of aircraft in NED coordinate system
    Vector3f velav;
    // only use if ahrs.have_inertial_nav() is true
    if (_ahrs.get_velocity_NED(velav)) {
    }
    else {Vector2f(velav.x, velav.y)=_ahrs.groundspeed_vector();
          /* if (gps.status() >= AP_GPS::GPS_OK_FIX_3D && gps.have_vertical_velocity()) {
                  velav = gps.velocity().z;
              } else {
                  velav = -barometer.get_climb_rate();
              }; */
          velav.z = 0;
    }

    // store aircraft velocity
    aircraft_vel = velav;

    const Vector2f velalv(velav.x,velav.y);
    const float velal = MAX(velalv.length(), 1.0f);

    // unit vector pointing from the center to the  aircraft
    Vector2f erlv;
    erlv = posalv;
    if (erlv.length() > 0.1f) {
        erlv = posalv.normalized();
    } else {
        if (velalv.length() < 0.1f) {
            erlv = Vector2f(cosf(_ahrs.yaw), sinf(_ahrs.yaw));
        } else {
            erlv = velalv.normalized();
        }
    }

    // Calculate time varying control parameters
    // Calculate the L1 length required for specified period
    // 0.3183099 = 1/pi
    _L1_dist = 0.3183099f * _L1_damping * _L1_period * velal;



    //Calculate Nu to capture center_WP
    const float xtrackVelCap = erlv % velalv; // Velocity across line - perpendicular to radial inbound to WP
    const float ltrackVelCap = - (velalv * erlv); // Velocity along line - radial inbound to WP
    float Nu = atan2f(xtrackVelCap,ltrackVelCap);

    _prevent_indecision(Nu);
    _last_Nu = Nu;

    Nu = constrain_float(Nu, -M_PI_2, M_PI_2); //Limit Nu to +- Pi/2

    //Calculate lat accln demand to capture center_WP (use L1 guidance law)
    const float latAccDemCap = K_L1 * velal * velal / _L1_dist * sinf(Nu);

    // calculate desired position on ellipse with major and minor principal axes along unit vectors e1 and e2, respectively
    // for given position vector posalv(phia) = ra(cos(phia)e1 + cos(theta)sin(phia)e2) of the aircraft
    //
    //hal.console->println(maxradius_cm);
    const float cos_psi = cosf(radians(psi));
    const float sin_psi = sinf(radians(psi));
    // trigonometric functions of angle at which a circle has to be inclined in order to yield the ellipse
    // poselv(phi) = cos(phi)e1 + cos(theta)sin(phi)e2
    const float cos_theta = minmaxratio;
    const float sin_theta = sqrt(1.0f - sq(minmaxratio));
    // unit vectors e1 and e2 into the direction of the major and minor principal axes, respectively
    const Vector2f e1(cos_psi,sin_psi);
//    hal.console->print("e1: ");
//    hal.console->print(e1.x);
//    hal.console->print(", ");
//    hal.console->println(e1.y);

    const Vector2f e2(-e1.y,e1.x);
    //hal.console->print(e1.x);
    //hal.console->print(" ");
    //hal.console->println(e1.y);
    // projections of the aircraft's position onto e1 and e2
    const float posal1 = posalv * e1;
    const float posal2 = posalv * e2;
	
//阿基米德螺线
	const float ra = sqrt(sq(posal1) + sq(1/cos_theta * posal2));
	/*if (ra<0)
		{
        maxradius= maxradius_cm;
	}
	else
		{
		maxradius= ra*100+_target_bearing_cd*M_PI/180;
	}*/
//sin正选曲线
	if (ra<0)
		{
        maxradius= maxradius_cm;
	}
	else
		{
		maxradius= ra*100 + ra * sin(2.0f*_target_bearing_cd*M_PI/360);
	}

	/*AP::logger().Write("Ehig", "TimeUS,x,y,po1,po2", "Qffff",
                                               AP_HAL::micros64(),
                                               (double)posav.x,
                                               (double)posav.y,
                                               (double)posal1,
                                               (double)posal2
                                               );*/
    // determine parametrization (ra,phia) of the aircraft's position from lateral components
    // posal1 = ra cos(phia)
    // posal2 = ra cos(theta)sin(phia);
    // projections of the aircraft's velocity onto e1 and e2
    //const float velal1 = velalv * e1;
    //const float velal2 = velalv * e2;

    // distance of the aircraft from the curve;
    float dae;
    // unit tangent vector
    Vector2f etelv;
    // unit outer normal vector at point of the ellipse closest to the aircraft's position relative to center_loc
    Vector2f enelv;
    // curvature at point of the ellipse closest to the aircraft's position relative to center_loc
    float kappa;

    //hal.console->print("cos_theta: ");
    // hal.console->print(cos_theta);
    if (!is_zero(cos_theta)){
      // non-degenerate ellipse
      // distance of the aircraft from the center of the ellipse in meter
      //const float ra = sqrt(sq(posal1) + sq(1/cos_theta * posal2));
	  //gcs().send_text(MAV_SEVERITY_INFO, "Tuning: Nu,ra %.2f %.2f",degrees(Nu),ra);
//      hal.console->print("ra: ");
//      hal.console->print(ra);
//           hal.console->println("cos_theta: ");
//           hal.console->println(cos_theta);
//           hal.console->println(S1ctoalv.x);
//           hal.console->println(S1ctoalv.y);
//           hal.console->println(e1.x);
//           hal.console->println(e1.y);
       const float rho = ra - maxradius/100.0f;
       // trigononometric functions of curve parameter phia at the aircraft's position
       const float cos_phia = posal1/ra;
       const float sin_phia = orientation * posal2/(ra * cos_theta);
       // first oder correction to curve parameter to approximate parameter at point of the ellipse closest to the aircraft's position
       const float dphi = - rho * sq(sin_theta) * sin_phia * cos_phia /(ra * (1 - sq(sin_theta * cos_phia)));
       const float cos_dphi = cosf(dphi);
       const float sin_dphi = sinf(dphi);
       // trigonometric functions of phi = phia + dphi, which is the first-order value of the curve parameter of the ellipse at the nearest point of the aircraft
       const float cos_phiapdphi = cos_phia * cos_dphi - sin_phia * sin_dphi;
       const float sin_phiapdphi = cos_phia * sin_dphi + sin_phia * cos_dphi;
       //hal.console->print(cos_phiapdphi);
       //hal.console->print(" ");
       //hal.console->println(sin_phiapdphi);
       // distance of the aircraft from the ellipse;
       dae = rho * cos_theta /sqrt(1 - sq(sin_theta * cos_phia));
       // position vector of point of the ellipse closest to the aircraft's position relative to center_loc
       // const Vector2f poselv = Vector2f((e1 * cos_phiapdphi + e2 * cos_theta * sin_phiapdphi * orientation) * maxradius_cm/100.0f);
       // projections onto e1 and e2
       //const float poselv1 = poselv * e1;
       //const float poselv2 = poselv * e2;
       const Vector2f telv = Vector2f(-e1 * sin_phiapdphi + e2 * cos_theta * cos_phiapdphi * orientation);
       const float telvnorm = telv.length();
       // unit tangent vector at point of the ellipse closest to the aircraft's position relative to center_loc
       etelv = telv / telvnorm;
       // unit outer normal vector at point of the ellipse closest to the aircraft's position relative to center_loc
       enelv(etelv.y * orientation, -etelv.x * orientation);
       // curvature at point of the ellipse closest to the aircraft's position relative to center_loc
       kappa = cos_theta /(ra * powf(telvnorm,3));
       //hal.console->print(cos_theta);
       //hal.console->print(" ");
       //hal.console->println(norm);

       //Calculate radial position and velocity errors
       const float xtrackVelCirc = enelv * velalv ; // normal outbound velocity
       const float xtrackErrCirc = dae; // Radial distance from the loiter circle
//       hal.console->print(xtrackVelCirc);
//       hal.console->print(", ");
//       hal.console->println(xtrackErrCirc);

       //hal.console->print("should be zero: ");
       //hal.console->println((posalv-poselv) * enelv - dae);
       const float ltrackVelCirc = etelv * velalv; // tangential velocity in the tangential direction (depends on orientation)

       // keep crosstrack error for reporting
       _crosstrack_error = xtrackErrCirc;
	   
	   //gcs().send_text(MAV_SEVERITY_INFO, "Tuning: rho,dae %.2f %.2f",(float)rho,(float)dae);
	   //gcs().send_text(MAV_SEVERITY_INFO, "Tuning: pia,dpi %.2f %.2f",(float)degrees(acosf(cos_phia)),(float)dphi);

	   AP::logger().Write("Elps", "TimeUS,ra,phia,phi,rhor,daer", "Qfffff",
                                               AP_HAL::micros64(),
                                               (double)ra,
                                               (double)degrees(acosf(cos_phia)),
                                               (double)degrees(acosf(cos_phiapdphi)),
                                               (double)rho,
                                               (double)dae
                                               );
	   
	   /*AP::logger().Write("Elps", "TimeUS,po1,po2,cosphia,dphi", "Qffff",
                                               AP_HAL::micros64(),
                                               (double)posal1,
                                               (double)posal2,
                                               (double)degrees(acosf(cos_phia)),
                                               (double)dphi
                                               );*/

       //Calculate PD control correction to circle waypoint_ahrs.roll
       float latAccDemCircPD = (xtrackErrCirc * Kx + xtrackVelCirc * Kv);

       //Calculate tangential velocity
       float velTangent = ltrackVelCirc; // * float(orientation);

//    //Prevent PD demand from turning the wrong way by limiting the command when flying the wrong way
//    if (ltrackVelCap < 0.0f && velTangent < 0.0f) {
//        latAccDemCircPD =  MAX(latAccDemCircPD, 0.0f);
//    }

       // Calculate centripetal acceleration demand
       float latAccDemCircCtr = velTangent * velTangent * kappa;
       //hal.console->print(velTangent);
       //hal.console->print("kappa: ");
       //hal.console->println(1/kappa);

       //Sum PD control and centripetal acceleration to calculate lateral manoeuvre demand
       float latAccDemCirc = orientation * (latAccDemCircPD + latAccDemCircCtr);
//       hal.console->println(latAccDemCircPD);

       // Perform switchover between 'capture' and 'circle' modes at the
       // point where the commands cross over to achieve a seamless transfer
       // Only fly 'capture' mode if outside the circle
       if ((xtrackErrCirc > maxradius * (1-minmaxratio)/100.0f && orientation * latAccDemCap < orientation * latAccDemCirc)){ //|| posalv.length()>= maxradius_cm/100.0f) {
           _latAccDem = latAccDemCap;
           _WPcircle = false;
           _bearing_error = Nu; // angle between demanded and achieved velocity vector, +ve to left of track
           _nav_bearing = atan2f(-posalv.y , -posalv.x); // bearing (radians) from AC to L1 point
           hal.console->println("capture");
       } else {
           _latAccDem = latAccDemCirc;
           _WPcircle = true;
           _bearing_error = 0.0f; // bearing error (radians), +ve to left of track
           _nav_bearing = atan2f(-posalv.y , -posalv.x); // bearing (radians)from AC to L1 point
//           hal.console->println("loiter_ellipse");
       }

       _data_is_stale = false; // status are correctly updated with current waypoint data

    } else {
        hal.console->println("degenerate ellipse: ");
        // if cos_theta == 0, the ellipse is degenerate; its lateral projection is a straight line spanned by e1

        Vector2f deltav = e1 * maxradius / 100.0f * 2.0 * orientation;
        struct Location start = center_loc;
        start.offset(-deltav.x, -deltav.y);
        struct Location end = center_loc;
        end.offset(deltav.x, deltav.y);
		//gcs().send_text(MAV_SEVERITY_INFO, "Tuning: s,e %.2f %.2f",start,end);

//        hal.console->print("deltav: ");
//        hal.console->print(deltav.x);
//        hal.console->print(", ");
//        hal.console->println(deltav.y);


        update_waypoint(start, end);
//        hal.console->print("_latAccDem: ");
//        hal.console->println(_latAccDem);


    }

    // current location of the aircraft
}


void AP_L1_Control::update_loiter_3d(const struct Location &S2center, const Vector3f &ercv, int32_t S2radius, const float &theta_r, int8_t orientation, struct Location &aircraft_loc, Vector3f &aircraft_vel, struct Location &desired_loc)
{

// CALCULATE DERIVED PARAMETERS FROM ARGUMENT LIST OF THE FUNCTION
    //int32_t  maxradius;
    const float cos_thetar = cosf(radians(theta_r));
    const float sin_thetar = sinf(radians(theta_r));
    // distance between the center of the sphere and the center of the circle
    //const int32_t D = S2radius * cos_thetar;
    const int32_t D = S2radius * cos_thetar - S2radius * cos_thetar + 1.0f;
    // radius of the circle
    const int32_t S1radius = S2radius * sin_thetar;
    // const Vector3f S2ctoS1cv = ercv * D/100.0f;
    // Location of the center of the circle
    struct Location S1center = S2center;
    S1center.offset(ercv.x * D/100.0f, ercv.y * D/100.0f);
    S1center.alt = S1center.alt - ercv.z * D + maxradius*0.5f ;//cos(pi/3)=0.5
     // trigonometric functions of the polar angle
     const float cos_theta = -ercv.z;
     const float sin_theta = sqrt(1.0f - sq(cos_theta));
      // calculate desired position on ellipse (= lateral projection of the circle) with major and minor principal axes along unit vectors e1 and e2, respectively
      // position vector of the aircraft parameterized as: posalv(phia) = ra(cos(phia)e1 + cos(theta)sin(phia)e2)
      // trigonometric functions of the azimuth angle
      float cos_psi;
      float sin_psi;
      // if theta == 0, i.e. the circle is horizontal, choose psi = 0
      if (is_zero(sin_theta)){
          cos_psi = 1;
          sin_psi = 0;
      } else {
          cos_psi = ercv.x / sin_theta;
          sin_psi = ercv.y / sin_theta;
      }
      // trigonometric functions of angle at which a circle has to be inclined in order to yield the ellipse as its lateral projection
      // unit vectors e1 and e2 into the direction of the major and minor principal axes, respectively
      //const Vector2f e1(cos_psi,sin_psi);
      //const Vector2f e2(-e1.y,e1.x);
      // minor and major principal axes directions have to be exchanged for the inclined circle
      // this can be accomplished by a 90 degree rotation: e1 -> e2, e2 -> -e1 which preserves the orientation
      const Vector2f e1(-sin_psi,cos_psi); //unit vector pointing along the major principal axis; directed towards east for psi = 0
      const Vector2f e2(-e1.y,e1.x);//unit vector pointing along the minor principal axis; directed towards south for psi = 0

	  // GET CURRENT POSITON AND VELOCITY
		   // current location of the aircraft
		   struct Location _current_loc;
	  
		   // get current position in NED coordinate system
		  if (_ahrs.get_position(_current_loc) == false) {
			  // if no GPS loc available, maintain last nav/target_bearing
			  _data_is_stale = true;
			  return;
		  }
		  // store aircraft's position for external use
		  aircraft_loc = _current_loc;
		  // aircraft's position vector (direction of ideal (straight) tether) from the center of the sphere
		  Vector3f S2ctoav(S2center.location_3d_diff_NED(_current_loc));
	  //	hal.console->print("S2ctoav: ");
	  //	hal.console->print(S2ctoav.x);
	  //	hal.console->print(", ");
	  //	hal.console->print(S2ctoav.y);
	  //	hal.console->print(", ");
	  //	hal.console->println(S2ctoav.z);
	  
	  
		  // aircraft's position vector from the center of the circle
		  Vector3f S1ctoav(S1center.location_3d_diff_NED(_current_loc));
		  // lateral projection of the aircraft's relative position vector
		  Vector2f S1ctoalv(S1ctoav.x, S1ctoav.y);
		  // update _target_bearing_cd
		  _target_bearing_cd = _current_loc.get_bearing_to(S1center);
	  
		  // track velocity in NED coordinate system
		  Vector3f velav;
		  // only use if ahrs.have_inertial_nav() is true
		  if (_ahrs.get_velocity_NED(velav)) {
		  }
		  else {Vector2f(velav.x, velav.y) =_ahrs.groundspeed_vector();
				/* if (gps.status() >= AP_GPS::GPS_OK_FIX_3D && gps.have_vertical_velocity()) {
						velav = gps.velocity().z;
					} else {
						velav = -barometer.get_climb_rate();
					}; */
				velav.z = 0;
		  }
		  // store aircraft velocity for external use
		  aircraft_vel = velav;
		  // lateral projection of the aircraft's velocity vector
		  Vector2f velalv(velav.x,velav.y);
		  // lateral velocity; protect against becoming zero
		  float velal = MAX(velalv.length(), 1.0f);
	  
	  
		  // unit tangent vector at point on a circle which is closest to the aircraft; lies in the plane containing the circle
		   Vector3f etv = (S1ctoav % ercv) * orientation;
		   etv = etv.normalized();
		   // lateral projection of the unit tangent vector
		   Vector2f etlv(etv.x, etv.y);
		   etlv = etlv.normalized();
	  
		   // outer unit normal (radial) vector at point on circle that is closest to the aircraft
		   Vector3f env = (ercv % etv) * orientation;
		   env = env.normalized(); // renormalize in order to compensate for numerical inaccuracies
		   // lateral renormalized projection of the unit normal vector; this is the radial vector of the point of the ellipse (the lateral projection of the circle) but NOT its normal vector;
		   Vector2f erlv(env.x, env.y);
		   if (erlv.length() > 0.1f) {
			   erlv = erlv.normalized();
		   } else {
			   if (velalv.length() < 0.1f) {
				   erlv = Vector2f(cosf(_ahrs.yaw), sinf(_ahrs.yaw));
			   } else {
				   erlv = velalv.normalized();
			   }
		   }


	 // projections of the aircraft's position onto e1 and e2
      const float posal1 = S1ctoalv * e1;
      const float posal2 = S1ctoalv * e2;

	  // distance of the aircraft from the center of the ellipse in meter
	  const float ra = sqrt(sq(posal1) + sq(1/cos_theta * posal2));
     //添加阿基米德螺线
     
	 if (ra<0)
		 {
		 maxradius = S1radius;
	 }
	 else
		 {
		 maxradius = ra*100.0f + _target_bearing_cd*M_PI/180;//add
		 //maxradius = ra*100.0f + sinf(_target_bearing_cd*M_PI/180);//add
		 gcs().send_text(MAV_SEVERITY_INFO, "Tuning: ra,radius,sin(b) %f,%f,%f",double(ra*100.0f),double(maxradius),double(sinf(_target_bearing_cd*M_PI/180)));
		 //maxradius = ra*100.0f-_target_bearing_cd*M_PI/180;//reduce
		// maxradius= ra*100+exp(_target_bearing_cd*M_PI/180*0.15f);
	 }
	  
      // minimal height for flying unconstrained outside the sphere
      int32_t heightmin_cm = 2000;
      // radius (half distance) between the two points of the segment lying above the minimal height
      int32_t segradius_cm = sqrt(sq(maxradius)-sq(heightmin_cm));
      //int32_t segradius_cm = sqrt(sq(S1radius)+sq(heightmin_cm));
      // vector is pointing in the direction of motion from start_loc to end_loc on the upper hemicircle (for inclination theta >0) given by orientation
      Vector2f maxv = - e1 * segradius_cm / 100.0f * orientation;
      struct Location start_loc = S1center;
      start_loc.offset(-maxv.x, -maxv.y);
      start_loc.alt = start_loc.alt + heightmin_cm;
      struct Location end_loc = S1center;
      end_loc.offset(maxv.x, maxv.y);
      end_loc.alt = end_loc.alt + heightmin_cm;
	  //gcs().send_text(MAV_SEVERITY_INFO, "Tuning: ser %.2f",(float)segradius_cm);
	  //gcs().send_text(MAV_SEVERITY_INFO, "Tuning: psi,ser %.2f %.2f",(float)cos_psi,(float)segradius_cm);
      // gcs().send_text(MAV_SEVERITY_INFO, "Tuning: s,e %.2f %.2f",(float)start_loc.alt,(float)end_loc.alt);
	  //gcs().send_text(MAV_SEVERITY_INFO, "Tuning: la,ln %.2f %.2f",(float)start_loc.lat,(float)start_loc.lng);
	  //gcs().send_text(MAV_SEVERITY_INFO, "Tuning: la,ln %.2f %.2f",(float)end_loc.lat,(float)end_loc.lng);



    // Calculate guidance gains used by PD loop (used during circle tracking)
      const float omega = (6.2832f / _L1_period);
      const float Kx = omega * omega;
      const float Kv = 2.0f * _L1_damping * omega;

	  //Qleaning
	if (_crosstrack_error<0.1f)
		{
		Lstate3d = 0;
	}
	else if(_crosstrack_error<0.1f && _crosstrack_error >-0.1f)
		{
		Lstate3d = 1;
	}
	else {
     Lstate3d = 2;
	}
	double Q[100][100]= {{0,0,500},
	                 {0,500,400},
	                 {0,0,500}};
	int best_action = inference_best_action(Lstate3d, Q);
	if (best_action ==0)
		{
		k3 = 3.8f;
	}
	else if(best_action ==1)
		{
		k3 = 4.0f;

	}
	else if(best_action == 2)
		{
		k3 = 6.0f;
	}
      // Calculate L1 gain required for specified damping (used during waypoint capture)
      const float K_L1 = k3 * _L1_damping * _L1_damping;
      //const float K_L1 = 4.0f * _L1_damping * _L1_damping;


// NAVIGATION ON THE LATERAL PROJECTION OF THE INCLINED CICRLE ((DEGENERATE) ELLIPSE)


      // projections of the aircraft's position onto e1 and e2
      /*const float posal1 = S1ctoalv * e1;
      const float posal2 = S1ctoalv * e2;*/
      // determine parametrization (ra,phia) of the aircraft's position from lateral components
      // posal1 = ra cos(phia)
      // posal2 = ra cos(theta)sin(phia);
      //gcs().send_text(MAV_SEVERITY_INFO, "Tuning: p1,p2 %.2f %.2f",(float)posal1,(float)posal2);
      // distance of the aircraft from the curve;
      float dae;
      // unit tangent vector
      Vector2f etelv;
      // unit outer normal vector at point of the ellipse closest to the aircraft's position relative to center_loc
      Vector2f enelv;
      // curvature at point of the ellipse closest to the aircraft's position relative to center_loc
      float kappa;

      //hal.console->print("cos_theta: ");
      // hal.console->print(cos_theta);
      if (!is_zero(cos_theta)){
          // non-degenerate ellipse
          // distance of the aircraft from the center of the ellipse in meter
          //const float ra = sqrt(sq(posal1) + sq(1/cos_theta * posal2));
//           hal.console->println("cos_theta: ");
//           hal.console->println(cos_theta);
//           hal.console->println(S1ctoalv.x);
//           hal.console->println(S1ctoalv.y);
//           hal.console->println(e1.x);
//           hal.console->println(e1.y);
          const float rho = ra - maxradius/100.0f;
          // trigonometric functions of curve parameter phia at the aircraft's position
          const float cos_phia = posal1/ra;
          const float sin_phia = orientation * posal2/(ra * cos_theta);
		  //gcs().send_text(MAV_SEVERITY_INFO, "Tuning: phi,rho %.2f %.2f",(float)degrees(acosf(cos_phia)),(float)rho);
          // first oder correction to curve parameter to approximate parameter at point of the ellipse closest to the aircraft's position
          const float dphi = - rho * sq(sin_theta) * sin_phia * cos_phia /(ra * (1 - sq(sin_theta * cos_phia)));
          const float cos_dphi = cosf(dphi);
          const float sin_dphi = sinf(dphi);
          // trigonometric functions of phi = phia + dphi, which is the first-order value of the curve parameter of the ellipse at the nearest point of the aircraft
          const float cos_phiapdphi = cos_phia * cos_dphi - sin_phia * sin_dphi;
          const float sin_phiapdphi = cos_phia * sin_dphi + sin_phia * cos_dphi;
          //hal.console->print(cos_phiapdphi);
          //hal.console->print(" ");
          //hal.console->println(sin_phiapdphi);
          // distance of the aircraft from the ellipse;
          dae = rho * cos_theta /sqrt(1 - sq(sin_theta * cos_phia));
          // position vector of point of the ellipse closest to the aircraft's position relative to center_loc
          // const Vector2f S1ctoelv = Vector2f((e1 * cos_phiapdphi + e2 * cos_theta * sin_phiapdphi * orientation) * S1radius/100.0f);
          // projections onto e1 and e2
          // const float S1ctoelv1 = S1ctoelv * e1;
          // const float S1ctoelv2 = S1ctoelv * e2;
          //hal.console->print(posel1);
          //hal.console->print(" ");
          //hal.console->println(posel2);
          const Vector2f telv = Vector2f(-e1 * sin_phiapdphi + e2 * cos_theta * cos_phiapdphi * orientation);
          const float telvnorm = telv.length();
          // unit tangent vector at point of the ellipse closest to the aircraft's position relative to center_loc
          etelv = telv / telvnorm;
          // unit outer normal vector at point of the ellipse closest to the aircraft's position relative to center_loc
          enelv = Vector2f(etelv.y * orientation, -etelv.x * orientation);
          // curvature at point of the ellipse closest to the aircraft's position relative to center_loc
          kappa = cos_theta /(ra * powf(telvnorm,3));

          // velocities and accelerations for capturing the center

          // crosstrack lateral velocity: velocity component of the aircraft along tangent vector  <=> velocity perpendicular to the path heading to the target point
          float xtrackVelCap = velalv * etlv * orientation;
          // along the track lateral velocity: velocity component of the aircraft radial inbound towards the target point
          float ltrackVelCap = - velalv * erlv;


          float Nu = atan2f(xtrackVelCap,ltrackVelCap);
          _prevent_indecision(Nu);
          _last_Nu = Nu;
          Nu = constrain_float(Nu, -M_PI_2, M_PI_2); //Limit Nu to +- Pi/2
          //Calculate lateral acceleration demand to capture center_WP (use L1 guidance law)
          float latAccDemCap = K_L1 * velal * velal / _L1_dist * sinf(Nu);


          // deviations of _current_loc from desired point
          // difference of the lengths of the direct lateral projections
          //float xtrackErrCirc = S2ctoalv.length() - S2ctocirclelv.length();
          // deviation of the position of the aircraft from the ellipse
          float xtrackErrCirc = dae;
          // keep crosstrack error for reporting
          _crosstrack_error = xtrackErrCirc;
		  //gcs().send_text(MAV_SEVERITY_INFO, "Tuning: err,u1 %.2f %.2f",(float)_crosstrack_error,(float)u1);
		  AP::logger().Write("L13d", "TimeUS,Err,po1,po2", "Qfff",
                                               AP_HAL::micros64(),
                                               (double)_crosstrack_error,
                                               (double)posal1,
                                               (double)posal2
                                               );

          // velocities and accelerations for circulating around the center

          //Vector2f veldeviationlv(velav.x,velav.y);
          // lateral velocity component in the direction of the outer normal vector
          float xtrackVelCirc = velalv * enelv;
          // lateral velocity component in the tangential direction of the ellipse
          float ltrackVelCirc = velalv * etelv;

          // calculate lateral acceleration for following the ellipse (the lateral projection of the circle)
          float latAccDemCircCtr = ltrackVelCirc * ltrackVelCirc * kappa;

          // calculate PD control correction to lateral acceleration
          // flight path outside desired circle -> positive correction
          // velocity pointing outwards -> positive correction
          float latAccDemCircPD = xtrackErrCirc * Kx + xtrackVelCirc * Kv;

          //Calculate tangential velocity
          float velTangent = xtrackVelCap * float(orientation);

          //Prevent PD demand from turning the wrong way by limiting the command when flying the wrong way
          if (ltrackVelCap < 0.0f && velTangent < 0.0f) {
              latAccDemCircPD =  MAX(latAccDemCircPD, 0.0f);
          }

          float tetherErr = 0;
          // sign of lateral acceleration corresponds to sign of roll angle
          // roll angle and hence acceleration is negative if orientation is positive
          float latAccDemCirc = orientation * (latAccDemCircPD + latAccDemCircCtr + tetherErr);

          // Perform switchover between 'capture' and 'circle' modes at the
          // point where the commands cross over to achieve a seamless transfer
          // Only fly 'capture' mode if outside the circle
          if (xtrackErrCirc > 0.0f && orientation * latAccDemCap < orientation * latAccDemCirc) {
              // capture mode
              // hal.console->println("non-degenerate case: capture");
              _latAccDem = latAccDemCap;
              _WPcircle = false;
              _bearing_error = Nu; // angle between demanded and achieved velocity vector, +ve to left of track
              _nav_bearing = 0;//atan2f(-erlv.y , -erlv.x); // bearing (radians) from AC to L1 point
              // desired target: location of point closest to the inclined circle
              desired_loc = S1center;
              desired_loc.offset(env.x * maxradius / 100.0f, env.y * maxradius / 100.0f);
              //desired_loc.alt = S1center.alt - env.z * maxradius;
			  desired_loc.alt = S1center.alt - env.z * maxradius;
              //hal.console->print("desired_loc.alt: ");
              //hal.console->println(desired_loc.alt);
          } else {
              // loiter
              // hal.console->println("loiter");
              _latAccDem = latAccDemCirc;
              _WPcircle = true;
              _bearing_error = 0.0f; // bearing error (radians), +ve to left of track
              _nav_bearing = atan2f(-erlv.y , -erlv.x); // bearing (radians)from AC to L1 point
              // desired target: point closest to the inclined circle
              desired_loc = S1center;
              desired_loc.offset(env.x * maxradius / 100.0f, env.y * maxradius / 100.0f);
              //desired_loc.alt = S1center.alt - env.z * maxradius;
			  desired_loc.alt = S1center.alt - env.z * maxradius;

              //hal.console->println(desired_loc.alt - S2center.alt);
              //hal.console->print(" ");
              //hal.console->println(S1center.alt-S2center.alt);

              // hal.console->print("height demanded: ");
              // hal.console->println(height);
          }
} else {
          //hal.console->println("degenerate case");
          // ellipse is degenerate to a line with unit tangent vector -e1, normal vector e2 running between start_loc and end_loc
          // aircraft should move along -e1 * orientation, i.e. in accord with the orientation of the upper (for theta = 90) hemicircle
          // e2, -e1, e_down form a right-handed coordinate system: looking downwards onto the (e1,e2)-plane in the -e1 direction, e2 points to the left
          // distance of the aircraft from the line
          // a deviation to the left requires a positive roll angle correction and hence positive acceleration
          // => deviation dae from path should be projection of the relative position onto e2 and should have sign given by orientation
          dae = posal2;
          etelv = -e1;
          enelv = e2;
//          hal.console->print("e1: ");
//          hal.console->print(e1.x);
//          hal.console->print(", ");
//          hal.console->println(e1.y);
//          hal.console->print("e2: ");
//          hal.console->print(e2.x);
//          hal.console->print(", ");
//          hal.console->println(e2.y);

          // determine point on the line closest to the aircraft
          // posal1 is projection of the aircraft's position onto the line: abs(posal1) > segradius_cm: projection lies outside segment
          //                                                          and   sgn(posal1) = +/-1: projection behind start point / in front of finish point
          struct Location capture_loc;
          // hal.console->print(posal1);
          // hal.console->print(", ");
          // hal.console->println(segradius_cm/100.0f);
          if (abs(posal1) >= segradius_cm/100.0f){
              if (posal1 >= 0) {
                  capture_loc = start_loc;
//                  hal.console->print("capture_loc = start_loc: ");
//                  hal.console->print(capture_loc.lat);
//                  hal.console->print(", ");
//                  hal.console->print(capture_loc.lng);
//                  hal.console->print(", ");
//                  hal.console->println(capture_loc.alt);
              } else {
                  capture_loc = end_loc;
//                  hal.console->print("capture_loc = end_loc: ");
//                  hal.console->print(capture_loc.lat);
//                  hal.console->print(", ");
//                  hal.console->print(capture_loc.lng);
//                  hal.console->print(", ");
//                  hal.console->println(capture_loc.alt);
              }
          } else {
              // int32_t heightsq = sq(S1radius)-sq(posal1* 100.0f);
              capture_loc = S1center;
              capture_loc.offset(e1.x * posal1, e1.y * posal1);
              capture_loc.alt = capture_loc.alt + sqrt(sq(S1radius)-sq(posal1* 100.0f));
//              hal.console->print("capture_loc = S1center + (");
//              hal.console->print(e1.x * posal1);
//              hal.console->print(", ");
//              hal.console->print(e1.y * posal1);
//              hal.console->print(", ");
//              hal.console->print(sqrt(sq(S1radius)-sq(posal1* 100.0f)));
//              hal.console->println(")");
//              hal.console->print(capture_loc.lat);
//              hal.console->print(", ");
//              hal.console->print(capture_loc.lng);
//              hal.console->print(", ");
//              hal.console->println(capture_loc.alt);
//              hal.console->print("capture_loc generic: ");
//              hal.console->print(e1.x * posal1);
//              hal.console->print(", ");
//              hal.console->println(e1.y * posal1);
//              hal.console->print("S1radius, posal1: ");
//              hal.console->print(S1radius);
//              hal.console->print(", ");
//              hal.console->println(posal1 * 100.0f);
          }






          // redefine radial vector in case of the degenerate ellipse as the vector on the line segment closest to the aircraft
          Vector2f rlv = capture_loc.get_distance_NE(_current_loc);
          erlv = rlv.normalized();
          // redefine tangential vector for approach to the target point (erlv, etlv, e_down) should form a right-handed system
          etlv = Vector2f(- erlv.y, erlv.x);

          // curvature of the straight line
          kappa = 0;

          // VELOCITIES AND ACCELERATIONS FOR CAPTURING THE CENTRAL POINT

          // crosstrack lateral velocity: velocity component of the aircraft along tangent vector  <=> velocity perpendicular to the path heading to the target point
          //  deviation should be positive if the inbound aircraft deviates towards left from the straight path to the target
          // (erlv, etlv) should form a right-handed system
          float xtrackVelCap = velalv * etlv * orientation;
          // along the track lateral velocity: velocity component of the aircraft radial inbound towards the target point
          float ltrackVelCap = - velalv * erlv;


          float Nu = atan2f(xtrackVelCap,ltrackVelCap);
          _prevent_indecision(Nu);
          _last_Nu = Nu;
          Nu = constrain_float(Nu, -M_PI_2, M_PI_2); //Limit Nu to +- Pi/2
          //Calculate lateral acceleration demand to capture center_WP (use L1 guidance law)
          float latAccDemCap = K_L1 * velal * velal / _L1_dist * sinf(Nu);


          // deviation of the position of the aircraft from the ellipse
          float xtrackErrCirc = dae;
          // keep crosstrack error for reporting
          _crosstrack_error = xtrackErrCirc;
		  gcs().send_text(MAV_SEVERITY_INFO, "Tuning: err %.2f",(float)_crosstrack_error);



          // update _target_bearing_cd
          _target_bearing_cd = _current_loc.get_bearing_to(end_loc);


          //Calculate Nu2 angle (angle of velocity vector relative to line connecting waypoints)
//                         xtrackVel = velalv % e1; // Velocity cross track
          float xtrackVelCirc = velalv * enelv;
//                         ltrackVel = velalv * e1;
          float ltrackVelCirc = velalv * etelv;
          float Nu2 = atan2f(xtrackVelCirc,ltrackVelCirc);
          //Calculate Nu1 angle (Angle to L1 reference point)
          float sine_Nu1 = _crosstrack_error / MAX(_L1_dist, 0.1f);
          //Limit sine of Nu1 to provide a controlled track capture angle of 45 deg
          sine_Nu1 = constrain_float(sine_Nu1, -0.7071f, 0.7071f);
          float Nu1 = asinf(sine_Nu1);
          float NuCirc = Nu1 + Nu2;
          _nav_bearing = atan2f(e1.y, e1.x) + Nu1; // bearing (radians) from AC to L1 point


          _prevent_indecision(NuCirc);
          _last_Nu = NuCirc;

          //Limit Nu to +-pi
          NuCirc = constrain_float(NuCirc, -1.5708f, +1.5708f);
          float latAccDemCirc = K_L1 * velal * velal / _L1_dist * sinf(NuCirc);

          // Waypoint capture status is always false during waypoint following
          _WPcircle = false;

          _bearing_error = NuCirc; // bearing error angle (radians), +ve to left of track


      // Perform switchover between 'capture' and 'circle' modes at the
      // point where the commands cross over to achieve a seamless transfer
      // Only fly 'capture' mode if outside the circle
          if (xtrackErrCirc > 0.0f && orientation * latAccDemCap < orientation * latAccDemCirc) {
              // capture mode
              // hal.console->println("degenerate case: capture");
              _latAccDem = latAccDemCap;
              _WPcircle = false;
              _bearing_error = Nu; // angle between demanded and achieved velocity vector, +ve to left of track
              _nav_bearing = 0;//atan2f(-erlv.y , -erlv.x); // bearing (radians) from AC to L1 point
              // desired target: location of point closest to the inclined circle
              desired_loc = S1center;
              desired_loc.offset(env.x * S1radius / 100.0f, env.y * S1radius / 100.0f);
              desired_loc.alt = S1center.alt - env.z * S1radius;
              //hal.console->print("desired_loc.alt: ");
              //hal.console->println(desired_loc.alt);
          } else {
              // loiter
              // hal.console->println("update_waypoint");
             _latAccDem = latAccDemCirc;
             _WPcircle = true;
             _bearing_error = 0.0f; // bearing error (radians), +ve to left of track
             _nav_bearing = atan2f(-erlv.y , -erlv.x); // bearing (radians)from AC to L1 point
              // desired target: point closest to the inclined circle
            desired_loc = capture_loc;
//            location_offset(desired_loc, env.x * S1radius / 100.0f, env.y * S1radius / 100.0f);
//            desired_loc.alt = S1center.alt - env.z * S1radius;
//            //hal.console->println(desired_loc.alt - S2center.alt);
              //hal.console->print(" ");
              //hal.console->println(S1center.alt-S2center.alt);

              // hal.console->print("height demanded: ");
              // hal.console->println(height);
          }
//          desired_loc = capture_loc;
//          hal.console->print("start_loc: ");
//           hal.console->print(start_loc.lat);
//           hal.console->print(", ");
//           hal.console->print(start_loc.lng);
//           hal.console->print(", ");
//           hal.console->println(start_loc.alt);
//           hal.console->print("desired_loc: ");
//          hal.console->print(desired_loc.lat);
//          hal.console->print(", ");
//          hal.console->print(desired_loc.lng);
//          hal.console->print(", ");
//          hal.console->println(desired_loc.alt);
//          hal.console->print("end_loc: ");
//          hal.console->print(end_loc.lat);
//          hal.console->print(", ");
//          hal.console->print(end_loc.lng);
//          hal.console->print(", ");
//          hal.console->println(end_loc.alt);
//          hal.console->print(location_diff(start_loc, end_loc).x);
//          hal.console->print(location_diff(start_loc, end_loc).y);
//
//
//
//          update_waypoint(start_loc, end_loc);
      }

}





void AP_L1_Control::update_loiter_3d(const struct Location &anchor, const struct Location &center_WP, float radius, float psi, float theta, float w, float sigma, int32_t dist, int8_t loiter_direction, Matrix3f M_pe, int32_t segment, struct Location &desired_loc)
{
    struct Location _current_loc;

    float cos_theta = cosf(theta);
    float sin_theta = sinf(theta);
    float cos_psi = cosf(psi);
    float sin_psi = sinf(psi);
    float cos_w = cosf(w);
    float sin_w = sinf(w);
    float cos_sigma = cosf(sigma);
    float sin_sigma = sinf(sigma);

    // Calculate guidance gains used by PD loop (used during circle tracking)
    float omega = (6.2832f / _L1_period);
    float Kx = omega * omega;
    float Kv = 2.0f * _L1_damping * omega;

    // Calculate L1 gain required for specified damping (used during waypoint capture)
    float K_L1 = 4.0f * _L1_damping * _L1_damping;

    // Get current position and velocity
    _ahrs.get_position(_current_loc);

    //position vector or rather tether
    Vector3f _position_vec_ef = anchor.location_3d_diff_NED(_current_loc);

    // track velocity in earth frame
    Vector3f _track_vel_ef;
   // _ahrs.get_velocity_NED(_track_vel_ef);

    Vector2f _groundspeed_vector;
    _groundspeed_vector.x = _track_vel_ef.x;
    _groundspeed_vector.y = _track_vel_ef.y;
    float groundSpeed = MAX(_groundspeed_vector.length() , 1.0f);

    Vector3f _track_vel_pf = M_pe * _track_vel_ef; // track velocity in plane frame
    Vector2f _track_vel; // projection of track velocity in xy plane of plane frame
    _track_vel.x = _track_vel_pf.x;
    _track_vel.y = _track_vel_pf.y;

    // Calculate the NED position of the aircraft relative to WP A
    Vector3f A_air_ef = center_WP.location_3d_diff_NED(_current_loc);
    Vector3f A_air_pf = M_pe * A_air_ef; // position of the aircraft in the plane frame

    Vector2f A_air; // projection of A_air_pf in xy plane of plane frame
    A_air.x = A_air_pf.x;
    A_air.y = A_air_pf.y;

    //hal.console->print(A_air.x);
    //hal.console->print(" ");
    //hal.console->println(A_air.y);
    // Calculate the unit vector from WP A to aircraft
    // protect against being on the waypoint and having zero velocity
    // if too close to the waypoint, use the velocity vector
    // if the velocity vector is too small, use the heading vector
    Vector2f A_air_unit;
    if (A_air.length() > 0.1f) {
        A_air_unit = A_air.normalized();
    } else {
        if (_track_vel.length() < 0.1f) {
            A_air_unit = Vector2f(cosf(_ahrs.yaw), sinf(_ahrs.yaw));
        } else {
            A_air_unit = _track_vel.normalized();
        }
    }

    // Calculate time varying control parameters
    // Calculate the L1 length required for specified period
    // 0.3183099 = 1/pi
    _L1_dist = 0.3183099f * _L1_damping * _L1_period * groundSpeed;
	gcs().send_text(MAV_SEVERITY_INFO, "Tuning: _L1_dist %.2f",_L1_dist);

    /*//Calculate the NE position of the aircraft relative to WP A
    Vector2f A_air_ef_2d = location_diff(center_WP, _current_loc);

    // Calculate the unit vector from WP A to aircraft
    // protect against being on the waypoint and having zero velocity
    // if too close to the waypoint, use the velocity vector
    // if the velocity vector is too small, use the heading vector
    Vector2f A_air_ef_2d_unit;
    if (A_air.length() > 0.1f) {
        A_air_ef_2d_unit = A_air_ef_2d.normalized();
    } else {
        if (_groundspeed_vector.length() < 0.1f) {
            A_air_ef_2d_unit = Vector2f(cosf(_ahrs.yaw), sinf(_ahrs.yaw));
        } else {
            A_air_ef_2d_unit = _groundspeed_vector.normalized();
        }
    }

    // update _target_bearing_cd

    _target_bearing_cd = get_bearing_cd(center_WP, _current_loc) - 100 * degrees(orientation);

    float cos_bearing = cosf(radians(_target_bearing_cd/100.0f));
    float sin_bearing = sinf(radians(_target_bearing_cd/100.0f));

    float bearing = radians(_target_bearing_cd/100.0f);
    float beta = atanf(cos_eta*cos_eta*tanf(bearing - M_PI/2));

    //Calculate Nu to capture center_WP
    float xtrackVelCap = (A_air_ef_2d_unit % _groundspeed_vector)* cosf(beta); // Velocity across line - perpendicular to radial inbound to WP
    float ltrackVelCap = - (_groundspeed_vector * A_air_ef_2d_unit)* cosf(beta); // Velocity along line - radial to ellipse
    float Nu = atan2f(xtrackVelCap,ltrackVelCap);

    //hal.console->println("beta");
    //hal.console->println(beta);
     */

    //hal.console->println("A Air x");
    //hal.console->println(A_air_unit.x);
    //hal.console->println("A Air y");
    //hal.console->println(A_air_unit.y);

    Vector3f Height_ef;         // calculate height of position on circle
    Height_ef.x = radius * A_air_unit.x;
    Height_ef.y = radius * A_air_unit.y;
    Height_ef.z = -dist/100.0f;
    Height_ef = M_pe.transposed() * Height_ef;

    //Calculate Nu to capture center_WP
    float xtrackVelCap = A_air_unit % _track_vel; // Velocity across line - perpendicular to radial inbound to WP
    float ltrackVelCap = -(_track_vel * A_air_unit); // Velocity along line - radial inbound to WP
    float Nu = atan2f(xtrackVelCap,ltrackVelCap);

    /*hal.console->println("groundspeed");
    hal.console->println(groundSpeed);
    hal.console->println("ltrackVelCap");
    hal.console->println(ltrackVelCap);
    hal.console->println("xtrackVelCap");
    hal.console->println(xtrackVelCap);*/

    _prevent_indecision(Nu);
    _last_Nu = Nu;

    Nu = constrain_float(Nu, -M_PI_2, M_PI_2); //Limit Nu to +- Pi/2

    //hal.console->println("Nu");
    //hal.console->println(Nu);

    //Calculate lat accln demand to capture center_WP (use L1 guidance law)
    float latAccDemCap = K_L1 * groundSpeed * groundSpeed / _L1_dist * sinf(Nu);

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    //Calculate errors for controlling
    //Calculate radial position and velocity errors and tether tension error
    //float xtrackVelCirc = -ltrackVelCap; // Radial outbound velocity - reuse previous radial inbound velocity
    //float xtrackErrCirc = A_air.length() - radius; // Radial distance from the loiter circle
    //float xtrackErrCirc = A_air_ef_2d.length() - radius*cos_eta/sqrtf(1-sin_eta*sin_eta*sin_bearing*sin_bearing);    // distance from circle projection (ellipse)
    //float v_unit_ef_x = loiter_direction*(-cos_eta*cos_orient*A_air_unit.y - sin_orient*A_air_unit.x); //M_ep * tangentialvector
    //float v_unit_ef_y = loiter_direction*(-cos_eta*sin_orient*A_air_unit.y + cos_orient*A_air_unit.x);
    float v_unit_ef_x = loiter_direction*(- A_air_unit.y*(cos_w*cos_sigma*cos_theta*cos_psi - sin_w*cos_theta*sin_psi - cos_w*sin_sigma*sin_theta)
                                          + A_air_unit.x*(-cos_w*cos_sigma*sin_psi - sin_w*cos_psi));   //M_ep * tangent vector
    float v_unit_ef_y = loiter_direction*(- A_air_unit.y*(sin_w*cos_sigma*cos_theta*cos_psi + cos_w*cos_theta*sin_psi - sin_w*sin_sigma*sin_theta)
                                          + A_air_unit.x*(-sin_w*cos_sigma*sin_psi + cos_w*cos_psi));
    float chi = atan2f(v_unit_ef_y, v_unit_ef_x);
    float sin_chi = sinf(chi);
    float cos_chi = cosf(chi);

    // outer normal vector in the lateral plane
    Vector2f v_n_lat(v_unit_ef_y, -v_unit_ef_x);
    // position vector of the aircraft relative to the center of the circle in the lateral plane
    Vector2f v_air_lat(A_air_ef.x, A_air_ef.y);


    // desired position vector on the circle relative to the center of the circle
    Vector3f radius_vec_pf;
    radius_vec_pf.x = radius * A_air_unit.x;
    radius_vec_pf.y = radius * A_air_unit.y;
    radius_vec_pf.z = 0;
    //float delta_height = dist/100.0f * cos_sigma*cos_theta - A_air_ef.z + Height_ef.z;  // cos_sigma*cos_theta = z component of normal_vec
    //Vector3f A_air_diff_pf = A_air_pf - radius_vec_pf;

    // desired lateral position vector on the lateral projection of the circle (the ellipse) relative to the center of the ellipse
    Vector3f v_ellipse_ef = M_pe.transposed()*radius_vec_pf;
    Vector2f v_ellipse_lat(v_ellipse_ef.x, v_ellipse_ef.y);
    // vector of the lateral deviation from current and desired position
    Vector2f A_diff_lat = v_air_lat - v_ellipse_lat;

    // sign of the deviation: +1 if aircraft is outside the circle, -1 if aircraft is inside the circle
    float sgn = v_n_lat * A_diff_lat;
    sgn = sgn/fabsf(sgn);



    float xtrackVelCirc = - sin_chi*_track_vel_ef.x + cos_chi*_track_vel_ef.y; // y Komponente von _track_vel_ef im bahnfesten Koordinatensystem
    //float xtrackVelCirc = -ltrackVelCap * cos_eta; // projection to xy plane of radial outbound velocity - reuse previous radial inbound velocity
//    int8_t sign;
//    //if((-sin_chi*cos_eta*cos_orient + cos_chi*cos_eta*sin_orient)*A_air_diff_pf.x +(sin_chi*sin_orient+cos_chi*cos_orient)*A_air_diff_pf.y +(-sin_chi*sin_eta*cos_orient+cos_chi*sin_eta*sin_orient)*A_air_diff_pf.z < 0) sign = -1;  //y componet of a_air_diff_kf
//    if (-sin_chi*(A_air_diff_pf.x * (cos_w*cos_sigma*cos_theta*cos_psi - sin_w*cos_theta*sin_psi - cos_w*sin_sigma*sin_theta)
//              + A_air_diff_pf.y * (-cos_w*cos_sigma*sin_psi - sin_w*cos_psi)
//              + A_air_diff_pf.z * (cos_w*cos_sigma*sin_theta*cos_psi - sin_w*sin_theta*sin_psi + cos_w*sin_sigma*cos_theta))
//
//      +cos_chi*(A_air_diff_pf.x * (sin_w*cos_sigma*cos_theta*cos_psi + cos_w*cos_theta*sin_psi - sin_w*sin_sigma*sin_theta)
//              + A_air_diff_pf.y * (-sin_w*cos_sigma*sin_psi + cos_w*cos_psi)
//              + A_air_diff_pf.z * (sin_w*cos_sigma*sin_theta*cos_psi + cos_w*sin_theta*sin_psi + sin_w*sin_sigma*cos_theta))
//              < 0) //y Komponente of A_air_diff im bahnfesten KS
//    {
//      sign = -1;
//    } else sign = 1;
//    //float xtrackErrCirc2 = sign * sqrtf(A_air_diff_pf.length()*A_air_diff_pf.length() - delta_height*delta_height); // distance from circle in xy plane
    float xtrackErrCirc = -sgn * A_diff_lat.length();


    float spring_const;
    float tether_length_demand;
    if (_position_vec_ef.length() > 390) {
        spring_const = 2;
    } else {
        spring_const = 0;
    }
    //float airspeed = _ahrs.get_airspeed()->get_airspeed();
    tether_length_demand = 400;// + 0.2*(airspeed - 30);


    Vector3f tether_tension = _position_vec_ef.normalized()*(spring_const*(tether_length_demand - 390)*4/7.785);
    float tetherErr = - sin_chi*tether_tension.x + cos_chi*tether_tension.y;

    //end of error calculation
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    /*hal.console->print("tether_length_demand: ");
    hal.console->println(tether_length_demand);
    hal.console->print("xtrackVelCirc: ");
    hal.console->println(-loiter_direction*xtrackVelCirc);
    hal.console->print("A_air_diff_pf.length() - delta height: ");
    hal.console->println(A_air_diff_pf.length() - delta_height);
    hal.console->print("delta_height angle stuff: ");
    hal.console->println(sin_sigma*sin_theta*sin_psi + cos_sigma*cos_theta);*/
    //hal.console->println(radius*cos_eta/sqrtf(1-sin_eta*sin_eta*sin_bearing*sin_bearing));

    // keep crosstrack error for reporting
   _crosstrack_error = xtrackErrCirc;


    //Calculate PD control correction to circle waypoint_ahrs.roll
    float latAccDemCircPD = -(xtrackErrCirc * Kx + xtrackVelCirc * Kv);

    //Calculate tangential velocity
    float velTangent = xtrackVelCap * float(loiter_direction);

    //hal.console->print("veltang: ");
    //hal.console->println(velTangent);
    //Prevent PD demand from turning the wrong way by limiting the command when flying the wrong way
    //if (ltrackVelCap < 0.0f && velTangent < 0.0f) {
    //    latAccDemCircPD =  MAX(latAccDemCircPD, 0.0f);
    //}

    // Calculate centripetal acceleration demand
    //float latAccDemCircCtr = velTangent * velTangent / MAX((0.5f * radius), (radius + xtrackErrCirc));
    //if(cos_eta < 0.05) cos_eta = 0.05;
    //float powerbasis = (cos_bearing*cos_bearing + cos_eta*cos_eta*cos_eta*cos_eta*sin_bearing*sin_bearing)/(1 - sin_eta*sin_eta*sin_bearing*sin_bearing);
    //float latAccDemCircCtr = velTangent * velTangent / MAX(0.5 * radius/cos_eta*sqrtf(pow(powerbasis,3)), ( radius/cos_eta*sqrtf(pow(powerbasis,3)) + xtrackErrCirc));
    //float latAccDemCircCtr = velTangent * velTangent / MAX(0.5 * radius, radius + xtrackErrCirc);
    //float latAccDemCircCtr = velTangent * velTangent / MAX(0.5 * radius, radius) * cos_clear / sqrtf(A_air_unit.y*A_air_unit.y*cos_eta*cos_eta + A_air_unit.x*A_air_unit.x); // y Komponente von a im bahnfesten Koordinatensystem
    float latAccDemCircCtr;
    if (segment % 2 == 0) {
        latAccDemCircCtr = velTangent * velTangent / MAX(0.5 * radius, radius)
            * cos_sigma*cos_theta / sqrtf(A_air_unit.y*A_air_unit.y*cos_theta*cos_theta + (-A_air_unit.y*sin_sigma*sin_theta + A_air_unit.x*cos_sigma)*(-A_air_unit.y*sin_sigma*sin_theta + A_air_unit.x*cos_sigma));
    }
    else {
        latAccDemCircCtr = velTangent * velTangent / MAX(0.5 * radius, radius)
                    * (-sin_sigma)*cos_psi / sqrtf(A_air_unit.x*A_air_unit.x*cos_psi*cos_psi + (A_air_unit.x*cos_sigma*sin_psi - A_air_unit.y*sin_sigma)*(A_air_unit.x*cos_sigma*sin_psi - A_air_unit.y*sin_sigma));
    }

    // correction for latAccDemCircCtr because of tether tension
/*    float latAccDemTether;
    if(_position_vec_ef.length() > 450) {
        latAccDemTether = 0.4/6*3*(_position_vec_ef.length()-450)*(A_air_ef*_position_vec_ef)/(A_air_ef.length()*_position_vec_ef.length());
    } else {
        latAccDemTether = 0;
    }
    hal.console->print("product air and pos: ");
    hal.console->println(A_air_ef*_position_vec_ef/(A_air_ef.length()*_position_vec_ef.length()));
*/
    //hal.console->print("a_lat: ");
    //hal.console->println(loiter_direction*latAccDemCircCtr);
    //hal.console->println("aProj");
    //hal.console->println(cos_eta / sqrtf(A_air_unit.y*A_air_unit.y*cos_eta*cos_eta + A_air_unit.x*A_air_unit.x));
    //hal.console->println("radius");
    //hal.console->println(radius);


    //hal.console->println("Kruemmungsradius");
    //hal.console->println(radius/cos_eta*sqrtf(pow(powerbasis,3)));

    //_nav_bearing = wrap_2PI(atan2f(A_air_unit.y , A_air_unit.x));


    //Sum PD control and centripetal acceleration to calculate lateral manoeuvre demand
    //float latAccDemCirc = loiter_direction * (latAccDemCircPD + latAccDemCircCtr) * sinf(acosf(-cosf(_nav_bearing)*sin_eta)); // projection to xy plane
    float latAccDemCirc = (latAccDemCircPD + loiter_direction * latAccDemCircCtr + tetherErr); // +tetherErr instead of -tetherErr because tethertension is showing from anchor to current position
//    float latAccDemCirc = (latAccDemCircPD + loiter_direction * latAccDemCircCtr);

    /*hal.console->print("latAccDemCirc: ");
    hal.console->println(loiter_direction*latAccDemCircCtr);
    hal.console->print("latAccDemPD: ");
    hal.console->println(latAccDemCircPD);*/

    // hal.console->println(sinf(acosf(-cosf(_nav_bearing)*sinf(M_PI/9))));
    // Perform switchover between 'capture' and 'circle' modes at the
    // point where the commands cross over to achieve a seamless transfer
    // Only fly 'capture' mode if outside the circle
    if (-loiter_direction*xtrackErrCirc > 0.0f && loiter_direction * latAccDemCap < loiter_direction * latAccDemCirc && false) {
        //height = dist * cos_sigma*cos_theta; //center_WP.alt - home.alt ;
        //hal.console->println("height cap");
        //hal.console->println(height);
        desired_loc = center_WP;
        _latAccDem = latAccDemCap;
        _WPcircle = false;
        _bearing_error = Nu; // angle between demanded and achieved velocity vector, +ve to left of track
        _nav_bearing = 0;
        //hal.console->println("nav_bearing");
        //hal.console->println(_nav_bearing);
        //hal.console->println("latAccDemCap");
        //hal.console->println(_latAccDem);
    } else {
        //height = -100*(Height_ef.z);
        //hal.console->print("height demanded: ");
        //hal.console->println(height);
        // determine the location of the desired point on the circle
        //hal.console->println(segment);
        desired_loc = center_WP;
        desired_loc.offset(v_ellipse_ef.x, v_ellipse_ef.y);
        //desired_loc.alt = dist * cos_sigma * cos_theta - 100.0f * v_ellipse_ef.z;
        desired_loc.alt = center_WP.alt - 100.0f * v_ellipse_ef.z;
        //hal.console->print("desired_loc.alt: ");
        //hal.console->println(desired_loc.alt);
        _latAccDem = latAccDemCirc; //-9.81/_position_vec_ef.z*sqrtf(_position_vec_ef.length_squared()-_position_vec_ef.z*_position_vec_ef.z);
        _WPcircle = true;
        _bearing_error = 0.0f; // bearing error (radians), +ve to left of track
        _nav_bearing = wrap_2PI(atan2f(A_air_unit.y , A_air_unit.x)); // bearing (radians)from AC to Minimum point of circle
        //hal.console->println(degrees(_nav_bearing));
        //hal.console->print("pitch: ");
        //hal.console->println(_ahrs.pitch_sensor/100.0f);
        //hal.console->println(nav_bearing_cd());
        //hal.console->println("latAccDemCirc");
        //hal.console->println(_latAccDem);
    }
    //    // Only fly 'capture' mode if outside the circle
    //    if (-loiter_direction*xtrackErrCirc > 0.0f && loiter_direction * latAccDemCap < loiter_direction * latAccDemCirc && false) {
    //        desired_loc = center_WP;
    //        // desired_loc.alt = dist * cos_sigma*cos_theta;
    //        //height = dist * cos_sigma*cos_theta; //center_WP.alt - home.alt ;
    //        //desired_loc = center_WP;
    //        //hal.console->println("height cap");
    //        //hal.console->println(height);
    //        hal.console->print("capture ");
    //        //hal.console->println(desired_loc.alt);
    ////        hal.console->println("capture");
    //        //hal.console->println(_nav_bearing);
    //        //hal.console->println("latAccDemCap");
    //        //hal.console->println(_latAccDem);
    //    } else {
    //        // height = -100*(Height_ef.z);
    //        // determine the location of the desired point on the circle
    //        desired_loc = center_WP;
    //        location_offset(desired_loc, v_ellipse_ef.x, v_ellipse_ef.y);
    //        //desired_loc.alt = dist * cos_sigma*cos_theta - 100.0f * v_ellipse_ef.z;
    //        desired_loc.alt = center_WP.alt - 100.0f * v_ellipse_ef.z;
    //        hal.console->print("eight_sphere: ");
    //        hal.console->println(desired_loc.alt);
    //        _latAccDem = latAccDemCirc; //-9.81/_position_vec_ef.z*sqrtf(_position_vec_ef.length_squared()-_position_vec_ef.z*_position_vec_ef.z);
    //        _WPcircle = true;
    //        _bearing_error = 0.0f; // bearing error (radians), +ve to left of track
    //        _nav_bearing = wrap_2PI(atan2f(A_air_unit.y , A_air_unit.x)); // bearing (radians)from AC to Minimum point of circle
    ////        hal.console->println("loiter_3d");
    //        //hal.console->print("pitch: ");
    //        //hal.console->println(_ahrs.pitch_sensor/100.0f);
    //        //hal.console->println(nav_bearing_cd());
    //        //hal.console->println("latAccDemCirc");
    //        //hal.console->println(_latAccDem);
    //    }
}


// update L1 control for heading hold navigation
void AP_L1_Control::update_heading_hold(int32_t navigation_heading_cd)
{
    // Calculate normalised frequency for tracking loop
    const float omegaA = 4.4428f/_L1_period; // sqrt(2)*pi/period
    // Calculate additional damping gain

    int32_t Nu_cd;
    float Nu;

    // copy to _target_bearing_cd and _nav_bearing
    _target_bearing_cd = wrap_180_cd(navigation_heading_cd);
    _nav_bearing = radians(navigation_heading_cd * 0.01f);

    Nu_cd = _target_bearing_cd - wrap_180_cd(_ahrs.yaw_sensor);
    Nu_cd = wrap_180_cd(Nu_cd);
    Nu = radians(Nu_cd * 0.01f);

    Vector2f _groundspeed_vector = _ahrs.groundspeed_vector();

    //Calculate groundspeed
    float groundSpeed = _groundspeed_vector.length();

    // Calculate time varying control parameters
    _L1_dist = groundSpeed / omegaA; // L1 distance is adjusted to maintain a constant tracking loop frequency
    float VomegaA = groundSpeed * omegaA;

    // Waypoint capture status is always false during heading hold
    _WPcircle = false;

    _crosstrack_error = 0;

    _bearing_error = Nu; // bearing error angle (radians), +ve to left of track

    // Limit Nu to +-pi
    Nu = constrain_float(Nu, -M_PI_2, M_PI_2);
    _latAccDem = 2.0f*sinf(Nu)*VomegaA;

    _data_is_stale = false; // status are correctly updated with current waypoint data
}

// update L1 control for level flight on current heading
void AP_L1_Control::update_level_flight(void)
{
    // copy to _target_bearing_cd and _nav_bearing
    _target_bearing_cd = _ahrs.yaw_sensor;
    _nav_bearing = _ahrs.yaw;
    _bearing_error = 0;
    _crosstrack_error = 0;

    // Waypoint capture status is always false during heading hold
    _WPcircle = false;

    _latAccDem = 0;

    _data_is_stale = false; // status are correctly updated with current waypoint data
}
