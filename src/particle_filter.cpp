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

	num_particles = 100;
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
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

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
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

	// particles in map coordinates
	// observations in vehicle coordinates

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
			obs_map.id = -1;
			observations_map.push_back(obs_map);
		}

		// each observation has the id landmark id assigned
		dataAssociation(landmarks_in_map, observations_map);

		double w = 1.0;
		double multiplier = 1.0;
		for (LandmarkObs observation : observations_map) {
			int id = observation.id;
			int index = id - 1;
			double dx = std::min(fabs(observation.x - landmarks_in_map[index].x), sensor_range);
			double dy = std::min(fabs(observation.y - landmarks_in_map[index].y), sensor_range);
			double std_x = std_landmark[0];
			double std_y = std_landmark[1];

			multiplier = 1.0/(2.0*M_PI*std_x*std_y)*exp( -0.5*(dx*dx/(std_x*std_x) + dy*dy/(std_y*std_y)) );
			if (multiplier == 0.0) {
				//std::cout << "id: " << id << " dx: " << dx << " dy: " << dy << std::endl;
			}
			w *= multiplier;
		}
		weights[i] = w;
	}

	for (double w : weights) {
		//std::cout << w << std::endl;
	}

	// normalize weights if
	double bias = 0.0;
	double weight_sum = std::accumulate(weights.begin(), weights.end(), bias);

	if (weight_sum != 0) {
		std::transform(weights.begin(), weights.end(), weights.begin(),
									 std::bind1st(std::multiplies<double>(), 1 / weight_sum));
	}

	// update particle weights
	for (int i = 0; i < particles.size(); i++) {
		Particle p = particles[i];
		p.weight = weights[i];
		particles[i] = p;
	}

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	std::default_random_engine generator;
	std::discrete_distribution<> disc_dist(weights.begin(), weights.end());
	std::vector<Particle> new_particles;

	while (new_particles.size() < num_particles) {
		int index = disc_dist(generator);
		Particle old_particle = particles[index];
		Particle new_particle;
		new_particle.x = old_particle.x;
		new_particle.y = old_particle.y;
		new_particle.theta = old_particle.theta;
		new_particle.id = -1;
		new_particles.push_back(new_particle);
	}
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
