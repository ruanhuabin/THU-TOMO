#include "track.h"

Track::Track()
{
    length=0;
}

Track::Track(float marker3D_x_new,float marker3D_y_new,float marker3D_z_new)
{
    length=0;
    marker3D_x=marker3D_x_new;
    marker3D_y=marker3D_y_new;
    marker3D_z=marker3D_z_new;
    status=true;
}

Track::~Track()
{

}

int Track::getLength()
{
    return length;
}

float Track::getMarker3D_x()
{
    return marker3D_x;
}

float Track::getMarker3D_y()
{
    return marker3D_y;
}

float Track::getMarker3D_z()
{
    return marker3D_z;
}

float Track::getMarker2D_x(int k)
{
    return marker2D_x[k];
}

float Track::getMarker2D_y(int k)
{
    return marker2D_y[k];
}

int Track::getMarker2D_n(int k)
{
    return n[k];
}

bool Track::getMarker2D_avail(int k)
{
    return marker2D_avail[k];
}

bool Track::getStatus()
{
    return status;
}

void Track::setMarker3D(float marker3D_x_new,float marker3D_y_new,float marker3D_z_new)
{
    marker3D_x=marker3D_x_new;
    marker3D_y=marker3D_y_new;
    marker3D_z=marker3D_z_new;
}

void Track::setMarker2D(float marker2D_x_new,float marker2D_y_new,int n_now)
{
    marker2D_x[n_now]=marker2D_x_new;
    marker2D_y[n_now]=marker2D_y_new;
}

void Track::addPatch(float marker2D_x_now,float marker2D_y_now,int n_now)
{
    length++;
    marker2D_x.push_back(marker2D_x_now);
    marker2D_y.push_back(marker2D_y_now);
    marker2D_avail.push_back(true);
    n.push_back(n_now);
}

void Track::changeStatus(bool status_now)
{
    status=status_now;
}

void Track::changeStatus2D(bool status_now,int n_now)
{
    marker2D_avail[n_now]=status_now;
}

void Track::move_origin(float dx,float dy,bool flip_x,bool flip_y)
{
    marker3D_x=marker3D_x+dx;
    if(flip_x)
    {
        marker3D_x=-marker3D_x;
    }
    marker3D_y=marker3D_y+dy;
    if(flip_y)
    {
        marker3D_y=-marker3D_y;
    }
    for(int t=0;t<length;t++)
    {
        marker2D_x[t]=marker2D_x[t]+dx;
        if(flip_x)
        {
            marker2D_x[t]=-marker2D_x[t];
        }
        marker2D_y[t]=marker2D_y[t]+dy;
        if(flip_y)
        {
            marker2D_y[t]=-marker2D_y[t];
        }
    }    
}
