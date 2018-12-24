/*
 * particle_filter.h
 *
 * 2D particle filter class.
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#ifndef PARTICLE_FILTER_H_
#define PARTICLE_FILTER_H_

#include <random>

#include "helper_functions.h"

struct Particle {

	int id;
	double x;
	double y;
	double theta;
	double weight;
	std::vector<int> associations;
	std::vector<double> sense_x;
	std::vector<double> sense_y;
};



class ParticleFilter {

private:
	// Number of particles to draw
	int num_particles;

	// Flag, if filter is initialized
	bool is_initialized;

	// Vector of weights of all particles
	std::vector<double> weights;

	// Random generator used for adding noise
	std::default_random_engine generator;

	std::vector<Map::single_landmark_s> findLandmarksWithinSensorRange(const Particle& p, const Map& map, double sensor_range);

	std::vector<Map::single_landmark_s> transformToMapCoordinates(const Particle& p, const std::vector<LandmarkObs>& observations);

	/*
	 * Set a particles list of associations, along with the associations calculated world x,y coordinates
	 * This can be a very useful debugging tool to make sure transformations are correct and assocations correctly connected
	 */
	void setAssociations(Particle& particle, const std::vector<Map::single_landmark_s>& associations);

	double calculateParticleWeight(const std::vector<Map::single_landmark_s>& predicted_landmarks,
			std::vector<Map::single_landmark_s>& observed_landmarks, double std_landmark[]);

	Map::single_landmark_s findClosestLandmark(const Map::single_landmark_s& observed_landmark,
			const std::vector<Map::single_landmark_s>& predicted_landmarks);
public:

	// Set of current particles
	std::vector<Particle> particles;

	// Constructor
	// @param n_particles Number of particles
	ParticleFilter(double n_particles) : num_particles(n_particles), is_initialized(false) {
		generator = std::default_random_engine();
	}

	// Destructor
	~ParticleFilter() {}

	/**
	 * init Initializes particle filter by initializing particles to Gaussian
	 *   distribution around first position and all the weights to 1.
	 * @param x Initial x position [m] (simulated estimate from GPS)
	 * @param y Initial y position [m]
	 * @param theta Initial orientation [rad]
	 * @param std[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
	 *   standard deviation of yaw [rad]]
	 */
	void init(double x, double y, double theta, double std[]);

	/**
	 * prediction Predicts the state for the next time step
	 *   using the process model.
	 * @param delta_t Time between time step t and t+1 in measurements [s]
	 * @param std_pos[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
	 *   standard deviation of yaw [rad]]
	 * @param velocity Velocity of car from t to t+1 [m/s]
	 * @param yaw_rate Yaw rate of car from t to t+1 [rad/s]
	 */
	void prediction(double delta_t, double std_pos[], double velocity, double yaw_rate);

	/**
	 * updateWeights Updates the weights for each particle based on the likelihood of the
	 *   observed measurements.
	 * @param sensor_range Range [m] of sensor
	 * @param std_landmark[] Array of dimension 2 [Landmark measurement uncertainty [x [m], y [m]]]
	 * @param observations Vector of landmark observations
	 * @param map Map class containing map landmarks
	 */
	void updateWeights(double sensor_range, double std_landmark[], const std::vector<LandmarkObs> &observations,
			const Map &map_landmarks);

	/**
	 * resample Resamples from the updated set of particles to form
	 *   the new set of particles.
	 */
	void resample();

	std::string getAssociations(Particle best);
	std::string getSenseX(Particle best);
	std::string getSenseY(Particle best);

	/**
	* initialized Returns whether particle filter is initialized yet or not.
	*/
	const bool initialized() const {
		return is_initialized;
	}
};



#endif /* PARTICLE_FILTER_H_ */
