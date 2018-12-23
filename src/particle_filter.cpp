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
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	for(int i=0; i<num_particles; i++) {
		Particle p = particles[i];
		std::vector<Map::single_landmark_s> predicted_landmarks = findLandmarksWithinSensorRange(p, map, sensor_range);
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

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

}

void ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
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
