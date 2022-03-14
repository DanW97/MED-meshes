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
        # South points
        vS = np.linspace(self.vmin, self.vmax, self.npts)
        uS = np.ones_like(vS)*self.umin - np.pi
        xS = xpts(uS, vS, self.scale[0], self.indices[0])
        yS = ypts(uS, vS, self.scale[1], self.indices[1])
        zS = zpts(vS, self.scale[2], self.indices[2])
        # West points
        vW = np.linspace(self.vmin, self.vmax, self.npts)
        uW = np.ones_like(vW)*self.umax -np.pi
        xW = xpts(uW, vW, self.scale[0], self.indices[0])
        yW = ypts(uW, vW, self.scale[1], self.indices[1])
        zW = zpts(vW, self.scale[2], self.indices[2])
        # North -> East points
        uNE = np.linspace(self.umin, self.umax, self.npts)
        vNE = np.zeros_like(uNE)
        xNE = xpts(uNE, vNE, self.scale[0], self.indices[0])
        yNE = ypts(uNE, vNE, self.scale[1], self.indices[1])
        zNE = zpts(vNE, self.scale[2], self.indices[2])
        # East -> South points
        uES = np.linspace(self.umin+np.pi/2, self.umax+np.pi/2, self.npts)
        vES = np.zeros_like(uES)
        xES = xpts(uES, vES, self.scale[0], self.indices[0])
        yES = ypts(uES, vES, self.scale[1], self.indices[1])
        zES = zpts(vES, self.scale[2], self.indices[2])
        # South -> West points
        uSW = np.linspace(self.umin+np.pi, self.umax+np.pi, self.npts)
        vSW = np.zeros_like(uSW)
        xSW = xpts(uSW, vSW, self.scale[0], self.indices[0])
        ySW = ypts(uSW, vSW, self.scale[1], self.indices[1])
        zSW = zpts(vSW, self.scale[2], self.indices[2])
        # West -> North points
        uWN = np.linspace(self.umin+np.pi*3/2, self.umax+np.pi*3/2, self.npts)
        vWN = np.zeros_like(uWN)
        xWN = xpts(uWN, vWN, self.scale[0], self.indices[0])
        yWN = ypts(uWN, vWN, self.scale[1], self.indices[1])
        zWN = zpts(vWN, self.scale[2], self.indices[2])
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
        # South spline
        Sspline_pts = []
        pts_append = Sspline_pts.append
        for i, (x, y, z) in enumerate(zip(xS, yS, zS)):
            pts_append(gm.addPoint(x,y,z,self.gr))
        Sspline_pts[-1] = Nspline_pts[-1] #this spline ends at the start of Nspline
        Sspline = gm.addSpline(Sspline_pts)
        # West spline
        Wspline_pts = [] 
        pts_append = Wspline_pts.append
        for i, (x, y, z) in enumerate(zip(xW, yW, zW)):
            pts_append(gm.addPoint(x,y,z,self.gr))
        Wspline_pts[-1] = Nspline_pts[-1] #this spline ends at the start of Nspline
        Wspline = gm.addSpline(Wspline_pts) 
        # North -> East spline
        NEspline_pts = [] #it starts from the end of the north spline
        pts_append = NEspline_pts.append
        for i, (x, y, z) in enumerate(zip(xNE, yNE, zNE)):
            pts_append(gm.addPoint(x,y,z,self.gr))
        NEspline_pts[0] = Nspline_pts[0] #it finishes at the start of the north spline    
        NEspline_pts[-1] = Espline_pts[0] #it finishes at the start of the east spline
        NEspline = gm.addSpline(NEspline_pts)
        """# East -> South spline
        ESspline_pts = [] #it starts from the end of the north spline
        pts_append = ESspline_pts.append
        for i, (x, y, z) in enumerate(zip(xES, yES, zES)):
            pts_append(gm.addPoint(x,y,z,self.gr))
        ESspline_pts[0] = Espline_pts[0] #it finishes at the start of the north spline    
        ESspline_pts[-1] = Sspline_pts[0] #it finishes at the start of the east spline
        ESspline = gm.addSpline(ESspline_pts)
        # South -> West spline
        SWspline_pts = [] #it starts from the end of the north spline
        pts_append = SWspline_pts.append
        for i, (x, y, z) in enumerate(zip(xSW, ySW, zSW)):
            pts_append(gm.addPoint(x,y,z,self.gr))
        SWspline_pts[0] = Sspline_pts[0] #it finishes at the start of the north spline    
        SWspline_pts[-1] = Wspline_pts[0] #it finishes at the start of the east spline
        SWspline = gm.addSpline(SWspline_pts)
        # West -> North spline
        WNspline_pts = [] #it starts from the end of the north spline
        pts_append = WNspline_pts.append
        for i, (x, y, z) in enumerate(zip(xWN, yWN, zWN)):
            pts_append(gm.addPoint(x,y,z,self.gr))
        WNspline_pts[0] = Wspline_pts[0] #it finishes at the start of the north spline    
        WNspline_pts[-1] = Nspline_pts[0] #it finishes at the start of the east spline
        WNspline = gm.addSpline(WNspline_pts)
        # Draw lower splines
        # North spline
        Nspline_pts_lower = []
        pts_append = Nspline_pts_lower.append
        for i, (x, y, z) in enumerate(zip(xN, yN, zN)):
            pts_append(gm.addPoint(x,y,-z,self.gr))
        Nspline_pts_lower[0] = NEspline_pts[0]
        Nspline_lower = gm.addSpline(Nspline_pts_lower)
        # East spline
        Espline_pts_lower = [] 
        pts_append = Espline_pts_lower.append
        for i, (x, y, z) in enumerate(zip(xE, yE, zE)):
            pts_append(gm.addPoint(x,y,-z,self.gr))
        Espline_pts_lower[0] = NEspline_pts[-1]
        Espline_pts_lower[-1] = Nspline_pts_lower[-1] #this spline ends at the start of Nspline
        Espline_lower = gm.addSpline(Espline_pts_lower) 
        # South spline
        Sspline_pts_lower = []
        pts_append = Sspline_pts_lower.append
        for i, (x, y, z) in enumerate(zip(xS, yS, zS)):
            pts_append(gm.addPoint(x,y,-z,self.gr))
        Sspline_pts_lower[0] = ESspline_pts[-1]
        Sspline_pts_lower[-1] = Nspline_pts_lower[-1] #this spline ends at the start of Nspline
        Sspline_lower = gm.addSpline(Sspline_pts_lower)
        # West spline
        Wspline_pts_lower = [] 
        pts_append = Wspline_pts_lower.append
        for i, (x, y, z) in enumerate(zip(xW, yW, zW)):
            pts_append(gm.addPoint(x,y,-z,self.gr))
        Wspline_pts_lower[0] = SWspline_pts[-1]
        Wspline_pts_lower[-1] = Nspline_pts_lower[-1] #this spline ends at the start of Nspline
        Wspline_lower = gm.addSpline(Wspline_pts_lower) """
        # Draw faces
        curve_loopNE = gm.addCurveLoop([-Nspline,NEspline,Espline])
        NEface = gm.addSurfaceFilling([curve_loopNE])
        #curve_loopES = gm.addCurveLoop([-Espline,ESspline,Sspline])
        #ESface = gm.addSurfaceFilling([curve_loopES])
        #curve_loopSW = gm.addCurveLoop([-Sspline,SWspline,Wspline])
        #SWface = gm.addSurfaceFilling([curve_loopSW])
        #curve_loopWN = gm.addCurveLoop([-Wspline,WNspline,Nspline])
        #WNface = gm.addSurfaceFilling([curve_loopWN])
        #curve_loopNE_lower = gm.addCurveLoop([-Nspline_lower,NEspline,Espline_lower], reorient=True)
        #NEface_lower = gm.addSurfaceFilling([curve_loopNE_lower])
        #curve_loopES_lower = gm.addCurveLoop([-Espline_lower,ESspline,Sspline_lower], reorient=True)
        #ESface_lower = gm.addSurfaceFilling([curve_loopES_lower])
        #curve_loopSW_lower = gm.addCurveLoop([-Sspline_lower,SWspline,Wspline_lower], reorient=True)
        #SWface_lower = gm.addSurfaceFilling([curve_loopSW_lower])
        #curve_loopWN_lower = gm.addCurveLoop([-Wspline_lower,WNspline,Nspline_lower], reorient=True)
        #WNface_lower = gm.addSurfaceFilling([curve_loopWN_lower])
        #sl1 = gm.addSurfaceLoop([NEface,NEface_lower,ESface,ESface_lower,SWface,SWface_lower,WNface,WNface_lower])
        sl1 = gm.addSurfaceLoop([NEface])
        v1 = gm.addVolume([sl1])
        gm.addPhysicalGroup(VOLUME, [v1])
        # Carve shape out of box defined by P1 = (-shell, -shell, -shell), P2 = (shell, shell, shell)
        """x = self.shell 
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
        points = []
        for pos in box:
            points.append(gm.addPoint(pos[0], pos[1], pos[2], self.gr))
        l1 = gm.addLine(points[0], points[1])
        l2 = gm.addLine(points[0], points[2])
        l3 = gm.addLine(points[0], points[4])
        l4 = gm.addLine(points[1], points[3])
        l5 = gm.addLine(points[1], points[5])
        l6 = gm.addLine(points[2], points[6])
        l7 = gm.addLine(points[2], points[3])
        l8 = gm.addLine(points[3], points[7])
        l9 = gm.addLine(points[4], points[5])
        l10 = gm.addLine(points[4], points[6])
        l11 = gm.addLine(points[5], points[7])
        l12 = gm.addLine(points[6], points[7])
        c1 = gm.add_curve_loop([l1,l4,-l7,-l2])
        c2 = gm.add_curve_loop([l5,l11,-l8,-l4]) 
        c3 = gm.add_curve_loop([-l9,l10,l12,-l11]) 
        c4 = gm.add_curve_loop([-l3,l2,l6,-l10]) 
        c5 = gm.add_curve_loop([l9,-l5,-l1,l3]) 
        c6 = gm.add_curve_loop([l7,l8,-l12,-l6])
        s1 = gm.add_surface_filling([c1])
        s2 = gm.add_surface_filling([c2])
        s3 = gm.add_surface_filling([c3])
        s4 = gm.add_surface_filling([c4])
        s5 = gm.add_surface_filling([c5])
        s6 = gm.add_surface_filling([c6])
        sl1 = gm.add_surface_loop([s1, s2, s3, s4, s5, s6])
        v1 = gm.add_volume([sl1, superquadric])
        gm.synchronize()
        gr1 = gm.addPhysicalGroup(
            SURFACE, [s1])
        gr2 = gm.addPhysicalGroup(
            SURFACE, [s3])
        gr3 = gm.addPhysicalGroup(
            SURFACE, [s2, s4, s5, s6])
        
        gr4 = gm.addPhysicalGroup(
            VOLUME, [v1]) """
        
        gm.synchronize()
        
        self.export()

    def export(self):
        """unclean way to add all the hard-coded stuff in, and rename the file"""
        gmsh.model.geo.synchronize()
        raw_file = f"{self.filepath}.geo_unrolled"
        new_file = f"{self.filepath}.geo"
        gmsh.write(raw_file)
        gmsh.model.mesh.setOrder(3)
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