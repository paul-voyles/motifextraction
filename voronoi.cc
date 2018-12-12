// Single Voronoi cell example code
//
// Author   : Chris H. Rycroft (LBL / UC Berkeley); Adapted by Jason Maldonis

#include <string>
#include <iostream>
#include <fstream>

#include "voro++.hh"
using namespace voro;

const double x_min=-1000,x_max=1000;
const double y_min=-1000,y_max=1000;
const double z_min=-1000,z_max=1000;
const int n_x=1,n_y=1,n_z=1;

// This function returns a random floating point number between 0 and 1
double rnd() {return double(rand())/RAND_MAX;}

int main(int argc, char *argv[]) {
    int i;
    double x,y,z;
	voronoicell v;

    int particles;
    std::string symbol;
    std::string comment;

    std::ifstream infile (argv[1]);
    if (!infile.is_open()) {
        std::cout << "Unable to open file"; 
        return -1;
    }

    infile >> particles;
    container con(x_min,x_max,y_min,y_max,z_min,z_max,n_x,n_y,n_z,false,false,false,particles);

    getline(infile, comment); // This gets the rest of the particles line (ie the newline)
    getline(infile, comment); // This gets the actual comment
    //std::cout << particles << '\n' << comment << '\n';
    for(i=0;i<particles;i++) {
        infile >> symbol;
        infile >> x;
        infile >> y;
        infile >> z;
        //std::cout << i << ' ' << x << ' ' << y << ' ' << z << '\n';
        getline(infile, comment);
        con.put(i,x,y,z);
    }
    infile.close();

    con.compute_cell(v, 0, particles-1);

    //printf("Face orders         : ");v.output_face_orders();puts("");
    v.output_face_orders();puts("");

}
