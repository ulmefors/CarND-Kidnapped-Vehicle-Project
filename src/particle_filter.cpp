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

#include "particle_filter.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {

	std::cout << "Initialize particle filter" << std::endl;

	num_particles = 40;
	double initial_weight = 1.0;

	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0.0, 1.0);

	for (int i = 0; i < num_particles; i++) {
		Particle p;
		p.id = -1;
		p.x = x + distribution(generator)*std[0];
		p.y = y + distribution(generator)*std[1];
		p.theta = theta + distribution(generator)*std[2];
		p.weight = initial_weight;
		particles.push_back(p);

		weights.push_back(initial_weight);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0.0, 1.0);

	for (int i = 0; i < particles.size(); i++) {
		// particle previous step (t-1)
		Particle old_p = particles[i];
		// predicted particle (t)
		Particle new_p;

		if (fabs(yaw_rate) < 0.001) {
			new_p.x = old_p.x + velocity*cos(old_p.theta)*delta_t;
			new_p.y = old_p.y + velocity*sin(old_p.theta)*delta_t;
		}
		else {
			new_p.x = old_p.x + velocity/yaw_rate*(sin(old_p.theta+yaw_rate*delta_t)-sin(old_p.theta));
			new_p.y = old_p.y + velocity/yaw_rate*(cos(old_p.theta)-cos(old_p.theta+yaw_rate*delta_t));
		}
		new_p.x += distribution(generator)*std_pos[0];
		new_p.y += distribution(generator)*std_pos[1];
		new_p.theta = old_p.theta + yaw_rate*delta_t + distribution(generator)*std_pos[2];

		// assign predicted particle to filter particles
		particles[i] = new_p;
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {

	for (int i = 0; i < observations.size(); i++) {
		// measurement observation
		LandmarkObs observation = observations[i];

		// distances between predicted observation and measurement observation
		std::vector<double> distances;
		for (LandmarkObs prediction : predicted) {
			distances.push_back(dist(observation.x, observation.y, prediction.x, prediction.y));
		}

		// index of minimum distance between predicted and observed landmark
		long min_index = std::distance(distances.begin(), std::min_element(distances.begin(), distances.end()));

		// assign landmark id to observation measurement
		observation.id = predicted[min_index].id;
		observations[i] = observation;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {

	// convert landmark_list to format comparable to observations
	std::vector<LandmarkObs> landmarks_in_map;
	for (Map::single_landmark_s landmark_in_map : map_landmarks.landmark_list) {
		LandmarkObs lmo;
		lmo.x = landmark_in_map.x_f;
		lmo.y = landmark_in_map.y_f;
		lmo.id = landmark_in_map.id_i;
		landmarks_in_map.push_back(lmo);
	}

	for (int i = 0; i < particles.size(); i++) {
		Particle p = particles[i];

		// convert observation to map coordinate (given particle p)
		std::vector<LandmarkObs> observations_map;
		for (LandmarkObs obs_veh : observations) {
			LandmarkObs obs_map;
			obs_map.x = obs_veh.x * cos(-p.theta) + obs_veh.y * sin(-p.theta) + p.x;
			obs_map.y = obs_veh.x * -sin(-p.theta) + obs_veh.y * cos(-p.theta) + p.y;
			obs_map.id = -1; // not used
			observations_map.push_back(obs_map);
		}

		// each observation is assigned closest landmark id
		dataAssociation(landmarks_in_map, observations_map);

		// assign weight using multi-variate Gaussian
		double w = 1.0;
		for (LandmarkObs observation : observations_map) {
			int id = observation.id;
			int index = id - 1; // landmark id start at 1, index at 0
			double dx = std::min(fabs(observation.x - landmarks_in_map[index].x), sensor_range);
			double dy = std::min(fabs(observation.y - landmarks_in_map[index].y), sensor_range);
			double std_x = std_landmark[0];
			double std_y = std_landmark[1];

			w *= 1.0/(2.0*M_PI*std_x*std_y)*exp( -0.5*(dx*dx/(std_x*std_x) + dy*dy/(std_y*std_y)) );
		}
		weights[i] = w;
	}

	// normalize weights
	double weight_sum = std::accumulate(weights.begin(), weights.end(), 0.0);
	if (weight_sum != 0) {
		std::transform(weights.begin(), weights.end(), weights.begin(),
									 std::bind1st(std::multiplies<double>(), 1 / weight_sum));
	}

	// update particle weights
	for (int i = 0; i < particles.size(); i++) {
		particles[i].weight = weights[i];
	}
}

void ParticleFilter::resample() {
	// random generator based on weight probabilities
	std::default_random_engine generator;
	std::discrete_distribution<> disc_dist(weights.begin(), weights.end());

	// generate new particles from weights
	std::vector<Particle> new_particles;
	while (new_particles.size() < num_particles) {
		int index = disc_dist(generator);
		Particle new_particle = particles[index];
		new_particles.push_back(new_particle);
	}
	// assign new particles to filter
	particles = new_particles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
