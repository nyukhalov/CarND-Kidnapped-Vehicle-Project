/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "helper_functions.h"
#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	std::cout << "Initializing particle filter. The particles number is " << num_particles << std::endl;

	std::normal_distribution<double> dist_x(x, std[0]);
	std::normal_distribution<double> dist_y(y, std[1]);
	std::normal_distribution<double> dist_theta(theta, std[2]);

	particles = std::vector<Particle>(num_particles);
	for(int i=0; i<num_particles; i++) {
		Particle p;
		p.x = dist_x(generator);
		p.y = dist_y(generator);
		p.theta = dist_theta(generator);
		particles[i] = p;
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	if (!is_initialized) {
		std::cout << "The filter must be initialized first" << std::endl;
		throw "Not initialized";
	}
	for(int i=0; i<num_particles; i++) {
		Particle p = particles[i];

		double theta = p.theta + yaw_rate * delta_t;
		double x = p.x + (velocity / yaw_rate) * (sin(theta) - sin(p.theta));
		double y = p.y + (velocity / yaw_rate) * (cos(p.theta) - cos(theta));

		std::normal_distribution<double> dist_x(x, std_pos[0]);
		std::normal_distribution<double> dist_y(y, std_pos[1]);
		std::normal_distribution<double> dist_theta(theta, std_pos[2]);

		p.x = dist_x(generator);
		p.y = dist_y(generator);
		p.theta = dist_theta(generator);
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map) {
	for(int i=0; i<num_particles; i++) {
		Particle p = particles[i];
		std::vector<Map::single_landmark_s> predicted_landmarks = findLandmarksWithinSensorRange(p, map, sensor_range);
		std::vector<Map::single_landmark_s> observed_landmarks = transformToMapCoordinates(p, observations);
		setAssociations(p, predicted_landmarks);
		p.weight = calculateParticleWeight(predicted_landmarks, observed_landmarks, std_landmark);
	}
}

std::vector<Map::single_landmark_s> ParticleFilter::findLandmarksWithinSensorRange(const Particle& p, const Map& map, double sensor_range) {
	std::vector<Map::single_landmark_s> predicted_landmarks;
	int n_landmarks = map.landmark_list.size();
	for(int i=0; i<n_landmarks; i++) {
		Map::single_landmark_s landmark = map.landmark_list[i];
		if (dist(p.x, p.y, landmark.x_f, landmark.y_f) <= sensor_range) {
			predicted_landmarks.push_back(landmark);
		}
	}
	return predicted_landmarks;
}

std::vector<Map::single_landmark_s> ParticleFilter::transformToMapCoordinates(const Particle& p, const std::vector<LandmarkObs>& observations) {
	std::vector<Map::single_landmark_s> t_observations;
	int size = observations.size();
	for(int i=0; i<size; i++) {
		LandmarkObs obs = observations[i];
		double x = cos(p.theta)*obs.x - sin(p.theta)*obs.y + p.x;
		double y = sin(p.theta)*obs.x + cos(p.theta)*obs.y + p.y;
		Map::single_landmark_s landmark;
		landmark.id_i = obs.id;
		landmark.x_f = x;
		landmark.y_f = y;
		t_observations.push_back(landmark);
	}
	return t_observations;
}

void ParticleFilter::setAssociations(Particle& particle, const std::vector<Map::single_landmark_s>& associations) {
	std::vector<int> ids;
	std::vector<double> xs;
	std::vector<double> ys;

	int size = associations.size();
	for (int i=0; i<size; i++) {
		Map::single_landmark_s landmark = associations[i];
		ids.push_back(landmark.id_i);
		xs.push_back(landmark.x_f);
		ys.push_back(landmark.y_f);
	}

	particle.associations = ids;
	particle.sense_x = xs;
	particle.sense_y = ys;
}

double ParticleFilter::calculateParticleWeight(const std::vector<Map::single_landmark_s>& predicted_landmarks,
			const std::vector<Map::single_landmark_s>& observed_landmarks, double std_landmark[]) {
	int size = observed_landmarks.size();
	double weight = 1.0f;
	for(int i=0; i<size; i++) {
		Map::single_landmark_s observed_landmark = observed_landmarks[i];
		Map::single_landmark_s closest_predicted_landmark = findClosestLandmark(observed_landmark, predicted_landmarks);
		double prob = guassianDistr(observed_landmark, closest_predicted_landmark, std_landmark);
		weight *= prob;
	}
	return weight;
}

Map::single_landmark_s ParticleFilter::findClosestLandmark(const Map::single_landmark_s& observed_landmark,
			const std::vector<Map::single_landmark_s>& predicted_landmarks) {
	return observed_landmark;
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
