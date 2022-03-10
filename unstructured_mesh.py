import gmsh 
import numpy as np
import sys
from scipy.optimize import minimize_scalar
from math import ceil

#Global constants - these form the "dim" section of the dimtag parlance used in gmsh
POINT = 0
CURVE = 1
SURFACE = 2
VOLUME = 3

#using parameterised functions from: https://en.wikipedia.org/wiki/Superquadrics
#auxilliary functions
def f(w, m): 
    return np.sign(np.sin(w))*np.power(np.abs(np.sin(w)), m)

def g(w, m):
    return np.sign(np.cos(w))*np.power(np.abs(np.cos(w)), m)

#parameterised functions that return make_points along a superquadric
# -pi/2<=v<=pi/2, -pi<=u<=pi
def xpts(u, v, scaleX, exponentX):
    return scaleX*g(v,2/exponentX)*g(u,2/exponentX)

def ypts(u, v, scaleY, exponentY):
    return scaleY*g(v,2/exponentY)*f(u,2/exponentY)

def zpts(v, scaleZ, exponentZ):
    return scaleZ*f(v,2/exponentZ)

#derivative of zpts function with respect to v, converted into an objective
#function to minimise
def diff_zpts(v, scaleZ, exponentZ):
    return (1 - (2*scaleZ*np.cos(v)*np.sin(v)**(2/exponentZ - 1))/exponentZ)**2

class SuperQuadric:
    def __init__(self, indices = [8, 8, 8], scale = [1, 1, 1], gr = 4, 
                filename = 'quad', rotation = [0, 0, 0], around = 10,
                near_sphere = 1.2) -> None:
        #checking
        assert len(scale) == 3
        assert len(rotation) == 3
        assert len(indices) == 3
        #survey the naughty list:
        if sum(np.array(indices)==2)>=2:
            self.ellipsoid = True
        else:
            self.ellipsoid = False
        if sum(np.array(rotation)!=0) > 0:
            self.rotatable = True 
        else:
            self.rotatable = False
        self.near_sphere = near_sphere
        self.around = around
        self.scale = scale 
        self.indices = indices 
        self.gr = gr
        self.vmax = np.pi/2
        res = minimize_scalar(diff_zpts, bounds = (0, np.pi/2), method='bounded', args = (scale[2], indices[2])) #explain this
        self.v = res.x
        self.u = np.pi/4
        self.filename = filename
        self.rotation = rotation
        self.shell = np.max(scale) + 2.5

    def fillet_radii(self):
        """
        Calculate fillet radii for each curved entity in the object

        Labelling scheme:

        5      7
        +------+.      
        |`. 1  | `.3    
        |  `+--+---+   following the below coordinate system: 
        |   |  |   |    
        +---+--+.  |   x\ |z
        6`. |  8 `.|     \|
           `+------+      +---y
            2      4

        """
        #create imaginary box to fillet
        x = self.scale[0]
        y = self.scale[1]
        z = self.scale[2]
        self.cube = np.array([
            [-x, -y,  z], #(1) -x -y  z
            [-x, -y, -z], #(2) -x -y -z
            [-x,  y,  z], #(3) -x  y  z
            [-x,  y, -z], #(4) -x  y -z
            [ x, -y,  z], #(5)  x -y  z
            [ x, -y, -z], #(6)  x -y -z
            [ x,  y,  z], #(7)  x  y  z
            [ x,  y, -z], #(8)  x  y -z
        ])
        self.quad_box = np.array([
            [xpts(self.u-np.pi, 0, self.scale[0], self.indices[0]), ypts(-self.u, 0, self.scale[1], self.indices[1]), zpts(self.v, self.scale[2], self.indices[2])],  #(1) -x -y  z
            [xpts(self.u-np.pi, 0, self.scale[0], self.indices[0]), ypts(-self.u, 0, self.scale[1], self.indices[1]), zpts(-self.v, self.scale[2], self.indices[2])], #(2) -x -y -z
            [xpts(self.u-np.pi, 0, self.scale[0], self.indices[0]), ypts(self.u, 0, self.scale[1], self.indices[1]),  zpts(self.v, self.scale[2], self.indices[2])],  #(3) -x  y  z
            [xpts(self.u-np.pi, 0, self.scale[0], self.indices[0]), ypts(self.u, 0, self.scale[1], self.indices[1]),  zpts(-self.v, self.scale[2], self.indices[2])], #(4) -x  y -z
            [xpts(self.u, 0, self.scale[0], self.indices[0]),       ypts(-self.u, 0, self.scale[1], self.indices[1]), zpts(self.v, self.scale[2], self.indices[2])],  #(5)  x -y  z
            [xpts(self.u, 0, self.scale[0], self.indices[0]),       ypts(-self.u, 0, self.scale[1], self.indices[1]), zpts(-self.v, self.scale[2], self.indices[2])], #(6)  x -y -z
            [xpts(self.u, 0, self.scale[0], self.indices[0]),       ypts(self.u, 0, self.scale[1], self.indices[1]),  zpts(self.v, self.scale[2], self.indices[2])],  #(7)  x  y  z
            [xpts(self.u, 0, self.scale[0], self.indices[0]),       ypts(self.u, 0, self.scale[1], self.indices[1]),  zpts(-self.v, self.scale[2], self.indices[2])], #(8)  x  y -z
        ])
        
        if not self.ellipsoid:
            #construct general superquadric that has a length + 2 fillet radii
            #for each point, we consider the planes xy and xz in that order
            quad_diagonals = [np.sqrt(self.quad_box[0,0]**2 + self.quad_box[0,1]**2), 
                            np.sqrt(self.quad_box[0,0]**2 + self.quad_box[0,2]**2)]
            #calculate the 16 fillet radii needed
            length = [x, y]
            diagonal = quad_diagonals[0]
            discriminant = (1-np.sum(length))**2 - (length[0]**2 + length[1]**2 - diagonal**2)
            self.radii_xy = ( -( (1-np.sum(length)) + np.sqrt(discriminant) ) )
            length = [x, z]
            diagonal = quad_diagonals[1]
            discriminant = (1-np.sum(length))**2 - (length[0]**2 + length[1]**2 - diagonal**2)
            self.radii_xz = ( -( (1-np.sum(length)) + np.sqrt(discriminant) ) )
            length = [x, z]
        else:       
            pass #TODO add elliptical code
            #construct ellipsoid which has >1 dimension that has a length of 2 fillet radii

    def draw(self):
        """
        Draw the superquadric

        Labelling scheme (the numbers represent each point_id):

        4      6
        +------+.      
        |`. 0  | `.2    
        |  `+--+---+   following the below coordinate system: 
        |   |  |   |    
        +---+--+.  |     x\ |z
        5`. |  7 `.|       \|
           `+------+   -y---+---y
            1      3        |\  
                         -z | \-x
                            
        NB: the actual point IDs are stored in dicts, these `points` are the base box
        points before filleting. This is because i am not good as ascii art!
        Planar Faces                  Curved Faces
        ------------                  ------------
        Face    Points      Name      ID      Start-Face  End-Face
        0       0, 1, 3, 2  Front     0       3           0 
        1       1, 5, 7, 3  Bottom    1       0           1
        2       5, 4, 6, 7  Back      2       1           2
        3       4, 0, 2, 6  Top       3       2           3
        4       4, 5, 1, 0  Left      4       0           4
        5       2, 3, 7, 6  Right     5       4           2 
                                      6       2           5
                                      7       5           0
                                      8       4           3
                                      9       3           5
                                      10      5           1
                                      11      1           4
        Rounded point_ids - X, Y and Z denote IDs of curve faces parallel with the respective axis that form the rounded point_id
        ---------------
        ID      X   Y   Z
        1       9   1   5
        2       12  2   5
        3       10  1   8
        4       11  2   8
        5       9   4   6
        6       12  3   6
        7       10  4   7
        8       11  3   7
        """
        if not self.ellipsoid:
            face_points = np.array([ 
                [0, 1, 3, 2],
                [1, 5, 7, 3],
                [5, 4, 6, 7],
                [4, 0, 2, 6],
                [4, 5, 1, 0],
                [2, 3, 7, 6]
            ])
            planes = np.array([
                [0, 1, 1],
                [1, 1, 0],
                [0, 1, 1],
                [1, 1, 0],
                [1, 0, 1],
                [1, 0, 1],
            ])
            self.fillet_radii()
            #dicts to store point IDs for each face for drawing to box edge
            self.face_points = {}
            self.face_lines = {}
            self.curved_lines = {}
            self.curve_loops = {}
            self.plane_faces = []
            self.curved_faces = []
            self.left = {}
            self.corner_ids = {}
            self.transfinite_lines = {}
            self.transfinite_surfaces = {}

            face_id = 0
            curve_id = 0
            point_id = 0
            #planar faces
            radii = np.array([self.radii_xz, self.radii_xz, self.radii_xy])
            geo = gmsh.model.geo
            msh = gmsh.model.mesh
            for i, (plane, face) in enumerate(zip(planes, face_points)):
                #draw points
                coords = self.cube[face, :]
                x = coords[:,0]-np.sign(coords[:,0])*plane[0]*radii[0]
                y = coords[:,1]-np.sign(coords[:,1])*plane[1]*radii[1]
                z = coords[:,2]-np.sign(coords[:,2])*plane[2]*radii[2]

                p1 = geo.add_point(x[0], y[0], z[0], self.gr)
                p2 = geo.add_point(x[1], y[1], z[1], self.gr)
                p3 = geo.add_point(x[2], y[2], z[2], self.gr)
                p4 = geo.add_point(x[3], y[3], z[3], self.gr)
                self.face_points[face_id] = [p1, p2, p3, p4]
                #draw lines
                l1 = geo.add_line(p1, p2)
                l2 = geo.add_line(p2, p3)
                l3 = geo.add_line(p3, p4)
                l4 = geo.add_line(p4, p1)  
                face_lines = [l1, l2, l3, l4]
                self.face_lines[face_id] = face_lines
                #for line in face_lines:
                    #geo.mesh.setTransfiniteCurve(line, ceil(self.around))
                #curve loops
                f1 = geo.add_curve_loop([l1, l2, l3, l4])
                self.curve_loops[face_id] = f1
                #surface
                s1 = geo.add_plane_surface([f1])
                self.plane_faces.append(s1)
                #geo.mesh.setTransfiniteSurface(s1)
                #geo.mesh.setRecombine(SURFACE, s1)
                face_id += 1
            #ellipse centres
            centers = [geo.add_point(*(point-np.sign(point)*radii), self.gr) for point in self.cube]
            #yz ellipses
            cen = np.array([0,1,5,4], dtype=int) #centers to iterate with
            start = np.array([3, 0, 1, 2], dtype=int)
            end = np.roll(start,3)
            for i in range(4): 
                l = geo.add_ellipse_arc(self.face_points[end[i]][0], centers[cen[i]], 
                                self.face_points[end[i]][0], self.face_points[start[i]][1])
                r = geo.add_ellipse_arc(self.face_points[start[i]][2], centers[cen[i]+2], 
                                self.face_points[start[i]][2],self.face_points[end[i]][3] )
                #geo.mesh.setTransfiniteCurve(l, ceil(self.around))
                #geo.mesh.setTransfiniteCurve(r, ceil(self.around))
                self.curved_lines[i] = [l, r]
                self.left[i] = [l]
            #xy ellipses
            cen = np.array([0, 4, 6, 2], dtype=int)
            start = np.array([0, 4, 2, 5], dtype=int)
            end = np.roll(start, 3)
            end_pt = [3,1,3,3] #because of the way it's drawn i can't perfectly maintain the pattern above
            for i in range(4): 
                if i == 2: #rotating point ids turned out to be a bit funky
                    l = geo.add_ellipse_arc(self.face_points[end[i]][end_pt[i]], centers[cen[i]], 
                                self.face_points[end[i]][end_pt[i]], self.face_points[start[i]][2],  )
                    r = geo.add_ellipse_arc(self.face_points[start[i]][3], centers[cen[i]+1], 
                                self.face_points[start[i]][3],self.face_points[end[i]][2] )
                else:
                    l = geo.add_ellipse_arc(self.face_points[end[i]][end_pt[i]], centers[cen[i]], 
                                self.face_points[end[i]][end_pt[i]], self.face_points[start[i]][0])
                    if i != 1:
                        r = geo.add_ellipse_arc(self.face_points[start[i]][1], centers[cen[i]+1], 
                                self.face_points[start[i]][1],self.face_points[end[i]][2] )
                    else:
                        r = geo.add_ellipse_arc(self.face_points[start[i]][1], centers[cen[i]+1], 
                                self.face_points[start[i]][1],self.face_points[end[i]][0] )
                #geo.mesh.setTransfiniteCurve(l, ceil(self.around))
                #geo.mesh.setTransfiniteCurve(r, ceil(self.around))
                self.curved_lines[i+4] = [l, r]
                self.left[i].append(l)
            #xz ellipses
            cen = np.array([0,2,3,1], dtype=int) #centers to iterate with
            start = np.array([4,3,5,1], dtype=int)
            end = np.roll(start,3)
            start_pt = [3,2,1,0] 
            end_pt = [1,0,3,2]
            for i in range(4): 
                if i == 1:
                    l = geo.add_ellipse_arc(self.face_points[end[i]][end_pt[i]], centers[cen[i]], 
                                    self.face_points[end[i]][end_pt[i]], self.face_points[start[i]][start_pt[i]])
                    r = geo.add_ellipse_arc(self.face_points[start[i]][3], centers[cen[i]+4], 
                                    self.face_points[start[i]][3],self.face_points[end[i]][3] )
                elif i < 3:
                    l = geo.add_ellipse_arc(self.face_points[end[i]][end_pt[i]], centers[cen[i]], 
                                    self.face_points[end[i]][end_pt[i]], self.face_points[start[i]][start_pt[i]])
                    r = geo.add_ellipse_arc(self.face_points[start[i]][i], centers[cen[i]+4], 
                                    self.face_points[start[i]][i],self.face_points[end[i]][end_pt[(i+1)%4]] )
                else:
                    l = geo.add_ellipse_arc(self.face_points[end[i]][end_pt[i]], centers[cen[i]], 
                                    self.face_points[end[i]][end_pt[i]], self.face_points[start[i]][start_pt[i]])
                    r = geo.add_ellipse_arc(self.face_points[start[i]][1], centers[cen[i]+4], 
                                    self.face_points[start[i]][1],self.face_points[end[i]][1] )
                #geo.mesh.setTransfiniteCurve(l, ceil(self.around))
                #geo.mesh.setTransfiniteCurve(r, ceil(self.around))
                self.curved_lines[i+8] = [l, r]
                self.left[i].append(l)
            #ellipse arc is start -> center -> point on major axis (start) -> end
            #create curved surfaces
            #start and end ids
            start = [3,0,1,2,0,4,2,5,4,3,5,1]
            end = [0,1,2,3,4,2,5,0,3,5,1,4]
            #line index for each face (where p1 - p4 are the points in order on the diagram)
            #l1 (p1, p2), l2 (p2, p3) l3 (p3, p4), l4 (p4, p1)
            tops    = [1]*4 + [0, 0, 2, 0, 3, 2, 1, 0, 0]
            bottoms = [3]*4 + [2, 0, 2, 2, 0, 3, 2, 1, 0]
            surfaces = []
            for i in range(12):
                loop = geo.add_curve_loop([self.face_lines[start[i]][tops[i]], self.curved_lines[i][1],
                                            self.face_lines[end[i]][bottoms[i]], self.curved_lines[i][0]])
                s = geo.add_surface_filling([loop])
                self.curved_faces.append(s)
                #geo.mesh.setTransfiniteSurface(s)
                #geo.mesh.set_recombine(SURFACE, s)
                surfaces.append(s)
            for face in self.plane_faces:
                surfaces.append(face)
            # now for the corners
            left = [
                [28, 45, 40],
                [26, 39, 43],
                [25, 41, 33],
                [34, 27, 47],
                ]
            right = [
                [46, 30, 38],
                [44, 37, 32],
                [42, 35, 31],
                [48, 29, 36],
                ]
            for i, (l, r) in enumerate(zip(left, right)):
                l_loop = geo.add_curve_loop([l[0], l[1], l[2]])
                l_s = geo.add_surface_filling([l_loop])
                #geo.mesh.setTransfiniteSurface(l_s)

                #geo.mesh.set_recombine(SURFACE, l_s)
                r_loop = geo.add_curve_loop([r[0], r[1], r[2]])
                r_s = geo.add_surface_filling([r_loop])
                #geo.mesh.setTransfiniteSurface(r_s)
                #geo.mesh.set_recombine(SURFACE, r_s)
                self.corner_ids[i] = [l_s, r_s]
            x = self.shell 
            y = self.shell 
            z = self.shell
            geo.synchronize()
            surface_list = gmsh.model.getEntitiesInBoundingBox(-x, -y, -z, x, y, z, dim = SURFACE)
            surfaces = [surface[1] for surface in surface_list]
            self.surface_loop = geo.add_surface_loop(surfaces)
            #v2 = geo.add_volume([self.surface_loop])
            geo.synchronize()
            if self.rotatable:
                self.rotate_obj()
            box = np.array([
            [-x, -y,  z], #(1) -x -y  z
            [-x, -y, -z], #(2) -x -y -z
            [-x,  y,  z], #(3) -x  y  z
            [-x,  y, -z], #(4) -x  y -z
            [ x, -y,  z], #(5)  x -y  z
            [ x, -y, -z], #(6)  x -y -z
            [ x,  y,  z], #(7)  x  y  z
            [ x,  y, -z], #(8)  x  y -z
            ])
            points = []
            for pos in box:
                points.append(geo.addPoint(pos[0], pos[1], pos[2], self.gr))
            geo.synchronize()
            l1 = geo.addLine(points[0], points[1])
            l2 = geo.addLine(points[0], points[2])
            l3 = geo.addLine(points[0], points[4])
            l4 = geo.addLine(points[1], points[3])
            l5 = geo.addLine(points[1], points[5])
            l6 = geo.addLine(points[2], points[6])
            l7 = geo.addLine(points[2], points[3])
            l8 = geo.addLine(points[3], points[7])
            l9 = geo.addLine(points[4], points[5])
            l10 = geo.addLine(points[4], points[6])
            l11 = geo.addLine(points[5], points[7])
            l12 = geo.addLine(points[6], points[7])
            c1 = geo.add_curve_loop([l1,l4,-l7,-l2])
            c2 = geo.add_curve_loop([l5,l11,-l8,-l4]) 
            c3 = geo.add_curve_loop([-l9,l10,l12,-l11]) 
            c4 = geo.add_curve_loop([-l3,l2,l6,-l10]) 
            c5 = geo.add_curve_loop([l9,-l5,-l1,l3]) 
            c6 = geo.add_curve_loop([l7,l8,-l12,-l6])
            s1 = geo.add_surface_filling([c1])
            s2 = geo.add_surface_filling([c2])
            s3 = geo.add_surface_filling([c3])
            s4 = geo.add_surface_filling([c4])
            s5 = geo.add_surface_filling([c5])
            s6 = geo.add_surface_filling([c6])
            sl1 = geo.add_surface_loop([s1, s2, s3, s4, s5, s6])
            v1 = geo.add_volume([sl1, self.surface_loop])
            geo.synchronize()
            #gmsh.model.mesh.embed(VOLUME, [v2], VOLUME, v1)
            #geo.synchronize()
          
            self.export()
                
        
        #else:
        #    pass #TODO figure out ellipsoid code - probably OCC revolve?

    def point_on_box(self, pos):
        """calculate the coordinates of a point on the box based on sign for the quadric point"""
        pos = np.array(pos)
        x = self.shell 
        y = self.shell 
        z = self.shell
        box = np.array([x,y,z])
        sign = np.sign(pos) 
        #guess which point lies on the box
        #try x
        dist = np.sqrt((pos-box)**2)
        pos[dist == min(dist)] = sign[dist == min(dist)]*self.shell
        return pos

    def rotate_obj(self): 
        x0 = 0
        y0 = 0
        z0 = 0
        ax = 0
        ay = 0
        az = 0
        #get every tag of quadric in box defined by scale size
        x = self.shell
        y = self.shell
        z = self.shell
        geo = gmsh.model.geo
        geo.synchronize()
        entities = gmsh.model.getEntitiesInBoundingBox(-x, -y, -z, x, y, z)

        for dim, angle in enumerate(self.rotation):
            if dim == 0 and angle != 0: #rotate around x-axis
                ax = 1
                geo.rotate(entities,x0,y0,z0,ax,ay,az,angle*np.pi/180)
            elif dim == 1 and angle != 0: #rotate around x-axis
                ay = 1
                geo.rotate(entities,x0,y0,z0,ax,ay,az,angle*np.pi/180)
            elif dim == 2 and angle != 0: #rotate around x-axis
                az = 1
                geo.rotate(entities,x0,y0,z0,ax,ay,az,angle*np.pi/180)

    def attach_in_box(self):
        """Join points to box of size 2Sx X 2Sy X 2Sz, centered at 0"""
        x = self.shell 
        y = self.shell 
        z = self.shell 
        box = np.array([
            [-x, -y,  z], #(1) -x -y  z
            [-x, -y, -z], #(2) -x -y -z
            [-x,  y,  z], #(3) -x  y  z
            [-x,  y, -z], #(4) -x  y -z
            [ x, -y,  z], #(5)  x -y  z
            [ x, -y, -z], #(6)  x -y -z
            [ x,  y,  z], #(7)  x  y  z
            [ x,  y, -z], #(8)  x  y -z
        ])
        if not self.ellipsoid:
            #collect all surfaces in box
            x = self.shell-0.5
            y = self.shell-0.5
            z = self.shell-0.5
            geo = gmsh.model.geo
            geo.synchronize()
            entities = gmsh.model.getEntitiesInBoundingBox(-x, -y, -z, x, y, z, dim = SURFACE)
            
            extension_dim = [0, 2, 0, 2, 1, 1]
            #the extension lines are rotationally invariant - limit rotation to +/- 45 deg
            volumes = []
            full_box = [0, 2]
            tall_box = [1, 3]
            #create a full box on faces 0 and 2 and just a tall one for faces 1 and 3
            #this should create enough curve loops to do a proper volume of the box
            import copy
            for i, (face, dim) in enumerate(zip(self.plane_faces, extension_dim)):
                if i in full_box:
                    face_bbox = gmsh.model.get_bounding_box(SURFACE, face)
                    line_points = gmsh.model.getEntitiesInBoundingBox(*face_bbox, POINT)
                    print(line_points)
                    x_box = [tag for _, tag in line_points]
                    y_box = []
                    z_box = []
                    for dim, tag in line_points:
                        point = gmsh.model.get_bounding_box(dim, tag)
                        point = np.array(point[0:3])
                        x_extreme = copy.deepcopy(point)
                        y_extreme = copy.deepcopy(point)
                        yz_extreme = copy.deepcopy(point)
                        xz_extreme = copy.deepcopy(point)
                        xy_extreme = copy.deepcopy(point)
                        z_extreme = point
                        x_extreme[0] = np.sign(x_extreme[0])*self.shell
                        y_extreme[1] = np.sign(y_extreme[1])*self.shell
                        z_extreme[2] = np.sign(z_extreme[2])*self.shell
                        yz_extreme[1:] = np.sign(yz_extreme[1:])*self.shell
                        xz_extreme[0] = np.sign(xz_extreme[0])*self.shell
                        xz_extreme[2] = np.sign(xz_extreme[2])*self.shell
                        xy_extreme[0] = np.sign(xy_extreme[0])*self.shell
                        xy_extreme[1] = np.sign(xy_extreme[1])*self.shell
                        x_box.append(geo.add_point(*x_extreme, self.gr))
                        y_box.append(geo.add_point(*y_extreme, self.gr))
                        z_box.append(geo.add_point(*z_extreme, self.gr))
                        geo.add_point(*yz_extreme, self.gr)
                        geo.add_point(*xz_extreme, self.gr)
                        geo.add_point(*xy_extreme, self.gr)
                    if i == 0: #hard coded lines
                        pass

                elif i in tall_box:
                    pass
                else:
                    surfaces = [face]
                    box_surface = [] #hold lines to form plane on box
                    for j, line in enumerate(self.face_lines[i]):
                        line_bbox = gmsh.model.get_bounding_box(CURVE, line)
                        line_points = gmsh.model.getEntitiesInBoundingBox(*line_bbox, POINT)
                        l1 = line #line on face of quadric
                        if j == 0:
                            p1 = line_points[1][1]
                            p4 = line_points[0][1]
                            p1_coords = gmsh.model.get_bounding_box(line_points[1][0], line_points[1][1])
                            p2_coords = np.array(p1_coords[0:3])
                            p2_coords[extension_dim[i]] = np.sign(p2_coords[extension_dim[i]])*self.shell
                            p2 = geo.add_point(*p2_coords, self.gr)
                            p4_coords = gmsh.model.get_bounding_box(line_points[0][0], line_points[0][1])
                            p3_coords = np.array(p4_coords[0:3])
                            p3_coords[extension_dim[i]] = np.sign(p3_coords[extension_dim[i]])*self.shell
                            p3 = geo.add_point(*p3_coords, self.gr)
                            end_point = p3 #save for end
                            l2 = geo.add_line(p1, p2)
                            l3 = geo.add_line(p2, p3)
                            l4 = geo.add_line(p3, p4)
                            end_line = l4 #save now for final curve surface
                            #l1 is already set from above
                            geo.mesh.set_transfinite_curve(l2, ceil(self.around), coef = self.near_sphere)
                            geo.mesh.set_transfinite_curve(l3, ceil(self.around), coef = self.near_sphere)
                            geo.mesh.set_transfinite_curve(l4, ceil(self.around), coef = self.near_sphere)
                            
                            c1 = geo.add_curve_loop([l1, l2, l3, l4])
                            
                        elif j == 1: #steal first created line from previous face
                            l2 = l2 #steal from previous
                            p2 = p2
                            p4_coords = gmsh.model.get_bounding_box(line_points[1][0], line_points[1][1])
                            p3_coords = np.array(p4_coords[0:3])
                            p3_coords[extension_dim[i]] = np.sign(p3_coords[extension_dim[i]])*self.shell
                            p3 = geo.add_point(*p3_coords, self.gr)
                            p4 = line_points[1][1]
                            l3 = geo.add_line(p2, p3)
                            l4 = geo.add_line(p3, p4)
                            #l1 is already set from above
                            geo.mesh.set_transfinite_curve(l2, ceil(self.around), coef = self.near_sphere)
                            geo.mesh.set_transfinite_curve(l3, ceil(self.around), coef = self.near_sphere)
                            geo.mesh.set_transfinite_curve(l4, ceil(self.around), coef = self.near_sphere)
                            
                            c1 = geo.add_curve_loop([-l1, l2, l3, l4])
                        
                        elif j == 2:
                            l2 = l4 #steal from previous
                            p2 = p3
                            p4_coords = gmsh.model.get_bounding_box(line_points[1][0], line_points[1][1])
                            p3_coords = np.array(p4_coords[0:3])
                            p3_coords[extension_dim[i]] = np.sign(p3_coords[extension_dim[i]])*self.shell
                            p3 = geo.add_point(*p3_coords, self.gr)
                            p4 = line_points[1][1]
                            l3 = geo.add_line(p2, p3)
                            l4 = geo.add_line(p3, p4)
                            #l1 is already set from above
                            
                            geo.mesh.set_transfinite_curve(l2, ceil(self.around), coef = self.near_sphere)
                            geo.mesh.set_transfinite_curve(l3, ceil(self.around), coef = self.near_sphere)
                            geo.mesh.set_transfinite_curve(l4, ceil(self.around), coef = self.near_sphere)
                            
                            c1 = geo.add_curve_loop([-l1, -l2, l3, l4])

                        elif j == 3:
                            l2 = l4 #steal from previous
                            p2 = p3
                            p4_coords = gmsh.model.get_bounding_box(line_points[0][0], line_points[0][1])
                            p3_coords = np.array(p4_coords[0:3])
                            p3_coords[extension_dim[i]] = np.sign(p3_coords[extension_dim[i]])*self.shell
                            p3 = geo.add_point(*p3_coords, self.gr)
                            p3 = end_point
                            p4 = line_points[1][1]
                            l3 = geo.add_line(p2, p3)
                            l4 = end_line
                            #l1 is already set from above
                            geo.mesh.set_transfinite_curve(l2, ceil(self.around), coef = self.near_sphere)
                            geo.mesh.set_transfinite_curve(l3, ceil(self.around), coef = self.near_sphere)
                            geo.mesh.set_transfinite_curve(l4, ceil(self.around), coef = self.near_sphere)
                            
                            c1 = geo.add_curve_loop([-l1, -l2, l3, l4])

                        s1 = geo.add_surface_filling([c1])
                        geo.mesh.set_transfinite_surface(s1)
                        geo.mesh.set_recombine(SURFACE, s1)
                        surfaces.append(s1)  
                        box_surface.append(l3) #add each l3 into a loop

                    l1 = box_surface[0]
                    l2 = box_surface[1]
                    l3 = box_surface[2]
                    l4 = box_surface[3]
                    c1 = geo.add_curve_loop([-l1, l2, l3, l4])
                    geo.add_curve_loop([])
                    s1 = geo.add_surface_filling([c1])
                    geo.mesh.set_transfinite_surface(s1)
                    geo.mesh.set_recombine(SURFACE, s1)
                    surfaces.append(s1)  
                    sl1 = geo.add_surface_loop(surfaces)
                    v1 = geo.add_volume([sl1])
                    geo.mesh.set_transfinite_volume(v1)
                    geo.mesh.set_recombine(VOLUME, v1)
                    volumes.append(v1)
            
            #curved faces - need to embed points on the box edges for the 
            
                      
                    





            #for surface in entities:
            #    dim, tag = surface
            #    face = gmsh.model.get_bounding_box(dim, tag)
            #    lines = gmsh.model.getEntitiesInBoundingBox(*face, dim = CURVE)
            #    for line in lines:
            #        dim, tag = line 
            #        bbox = gmsh.model.get_bounding_box(dim, tag)
            #        points = gmsh.model.getEntitiesInBoundingBox(*bbox, dim = POINT)
            #        box_points = []
            #        for point in points:
            #            dim, tag = point 
            #            coords = gmsh.model.getBoundingBox(dim, tag)
            #            pos = coords[0:3] #it's a point so the full box isn't needed
            #            box_point_coords = self.point_on_box(pos)
            #            box_point = geo.add_point(*box_point_coords, self.gr)
            #            box_points.append(box_point)
            #        l1 = tag
            #        l2 = geo.add_line(points[1][1], box_points[1]) 
            #        l3 = geo.add_line(box_points[1], box_points[0]) 
            #        l4 = geo.add_line(box_points[0], points[0][1])
            #        return l1










    def export(self):
        """unclean way to add all the hard-coded stuff in, and rename the file"""
        gmsh.model.geo.synchronize()
        raw_file = f"{self.filename}.geo_unrolled" 
        new_file = f"{self.filename}.geo" 
        gmsh.write(raw_file)
        gmsh.write(f"{self.filename}.msh")
        import os 
        os.system(f"mv {raw_file} {new_file}")
    
    def view(self):
        """
        Make gmsh open up and view superquadric
        """
        import os
        os.system(f"gmsh {self.filename}.geo")
        
gmsh.initialize(sys.argv) #initialise gmsh
gmsh.model.add("Superquadric") #add model
quad = SuperQuadric(
                   filename='uns_quad',
                   rotation=[45, 2, 50]
                     )
quad.draw()
quad.view() 