import enum
import gmsh
import numpy as np
import sys

# Global constants - these form the "dim" section of the dimtag parlance used in gmsh
POINT = 0
CURVE = 1
SURFACE = 2
VOLUME = 3

# using parameterised functions from: https://en.wikipedia.org/wiki/Superquadrics
# auxilliary functions


def f(w, m):
    return np.sign(np.sin(w))*np.power(np.abs(np.sin(w)), m)


def g(w, m):
    return np.sign(np.cos(w))*np.power(np.abs(np.cos(w)), m)

# parameterised functions that return make_points along a superquadric
# -pi/2<=v<=pi/2, -pi<=u<=pi


def xpts(u, v, scaleX, exponentX):
    return scaleX*g(v, 2/exponentX)*g(u, 2/exponentX)


def ypts(u, v, scaleY, exponentY):
    return scaleY*g(v, 2/exponentY)*f(u, 2/exponentY)


def zpts(v, scaleZ, exponentZ):
    return scaleZ*f(v, 2/exponentZ)


class Superquadric:
    def __init__(self, indices=[8, 8, 8], scale=[1, 1, 1],
                 filepath='quad', rotation=[0, 0, 0], npts = 20, gr = 1) -> None:
        # checking
        assert len(scale) == 3
        assert len(rotation) == 3
        assert len(indices) == 3
        if sum(np.array(rotation) != 0) > 0:
            self.rotatable = True
        else:
            self.rotatable = False
        self.scale = scale
        self.indices = indices
        self.rotation = rotation
        self.umin = np.pi/2
        self.umax = np.pi
        self.vmin = 0
        self.vmax = np.pi/2
        self.filepath = filepath
        self.shell = np.max(scale) + 2.5
        self.npts = npts
        self.gr = gr

    def draw(self):
        """Draw the superquadric object carved out of a box.

        Construct 1/8 of the object, and form the rest by copies. The section
        drawn is in the domain {x>0, y>0, z>0}

        The following conventions are used to define the orientation of the splines
        formed:
        1) N - the vertical spline parallel with x
        2) E - the vertical spline parallel with y
        3) NE - the horizontal spline parallel to the xy plane
        """
        gmsh.initialize(sys.argv)
        gmsh.model.add("Superquadric")
        gm = gmsh.model.geo  # "with" doesn't work
        # Define coordinates needed for the 3 spline lines
        # North points
        vN = np.linspace(self.vmin, self.vmax, self.npts)
        uN = np.ones_like(vN)*self.umin
        xN = xpts(uN, vN, self.scale[0], self.indices[0])
        yN = ypts(uN, vN, self.scale[1], self.indices[1])
        zN = zpts(vN, self.scale[2], self.indices[2])
        # East points
        vE = np.linspace(self.vmin, self.vmax, self.npts)
        uE = np.ones_like(vE)*self.umax
        xE = xpts(uE, vE, self.scale[0], self.indices[0])
        yE = ypts(uE, vE, self.scale[1], self.indices[1])
        zE = zpts(vE, self.scale[2], self.indices[2])
        # North -> East points
        uNE = np.linspace(self.umin, self.umax, self.npts)
        vNE = np.zeros_like(uNE)
        xNE = xpts(uNE, vNE, self.scale[0], self.indices[0])
        yNE = ypts(uNE, vNE, self.scale[1], self.indices[1])
        zNE = zpts(vNE, self.scale[2], self.indices[2])
        # Draw splines
        # North spline
        Nspline_pts = []
        pts_append = Nspline_pts.append
        for i, (x, y, z) in enumerate(zip(xN, yN, zN)):
            pts_append(gm.addPoint(x,y,z,self.gr))
        Nspline = gm.addSpline(Nspline_pts)
        # East spline
        Espline_pts = [] 
        pts_append = Espline_pts.append
        for i, (x, y, z) in enumerate(zip(xE, yE, zE)):
            pts_append(gm.addPoint(x,y,z,self.gr))
        Espline_pts[-1] = Nspline_pts[-1] #this spline ends at the start of Nspline
        Espline = gm.addSpline(Espline_pts) 
        # North -> East spline
        NEspline_pts = [] #it starts from the end of the north spline
        pts_append = NEspline_pts.append
        for i, (x, y, z) in enumerate(zip(xNE, yNE, zNE)):
            pts_append(gm.addPoint(x,y,z,self.gr))
        NEspline_pts[0] = Nspline_pts[0] #it finishes at the start of the north spline    
        NEspline_pts[-1] = Espline_pts[0] #it finishes at the start of the east spline
        NEspline = gm.addSpline(NEspline_pts)
        curve_loop = gm.addCurveLoop([Nspline,NEspline,Espline], reorient=True)
        NEface = gm.addSurfaceFilling([curve_loop])
        # Copy and rotate NEface to form whole object
        faces = [NEface]
        angle = [np.pi/2, np.pi, np.pi*3/2] #rotation angle
        origin = [0,0,0]
        axis = [0,0,1]
        for i in range(3): #7 faces to create
            copied_face = gm.copy([(SURFACE, NEface)])
            gm.rotate(copied_face, *origin, *axis, angle[i])
            faces.append(copied_face[0][1]) 
        NEface_lower = gm.copy([(SURFACE, NEface)])   
        gm.rotate(NEface_lower, *origin, 1, 0, 0, angle[1]) #flip to -z  
        faces.append(NEface_lower[0][1])
        for i in range(3): #7 faces to create
            copied_face = gm.copy(NEface_lower)
            gm.rotate(copied_face, *origin, *axis, angle[i])
            faces.append(copied_face[0][1])
        # Carve shape out of box defined by P1 = (-shell, -shell, -shell), P2 = (shell, shell, shell)
        
        gm.synchronize()
        self.export()

    def export(self):
        """unclean way to add all the hard-coded stuff in, and rename the file"""
        gmsh.model.geo.synchronize()
        raw_file = f"{self.filepath}.geo_unrolled"
        new_file = f"{self.filepath}.geo"
        gmsh.write(raw_file)
        gmsh.model.mesh.generate(VOLUME)
        gmsh.model.geo.synchronize()
        gmsh.write(f"{self.filepath}.msh")
        import os
        os.system(f"mv {raw_file} {new_file}")

    def view(self):
        """Visualise the geo file"""
        if '-nopopup' not in sys.argv:
            gmsh.fltk.run()

quad = Superquadric()
quad.draw()
quad.view()