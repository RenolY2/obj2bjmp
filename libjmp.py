from struct import unpack, pack
from math import ceil, inf, acos, degrees
from vectors import Vector3, Triangle, Vector2, Matrix3x3
from re import match
UPVECTOR = Vector3(0.0, 1.0, 0.0)
FWVECTOR = Vector3(1.0, 0.0, 0.0)
SIDEVECTOR = Vector3(0.0, 0.0, 1.0)


def round_vector(vector, digits):
    vector.x = round(vector.x, digits)
    vector.y = round(vector.y, digits)
    vector.z = round(vector.z, digits)


def read_vertex(v_data):
    split = v_data.split("/")
    if len(split) == 3:
        vnormal = int(split[2])
    else:
        vnormal = None
    v = int(split[0])
    return v#, vnormal


def read_uint32(f):
    val = f.read(0x4)
    return unpack(">I", val)[0]


def read_float_tripple(f):
    val = f.read(0xC)
    return unpack(">fff", val)


def read_vector3(f):
    xyz = unpack(">fff", f.read(0xC))
    return Vector3(*xyz)


def read_float(f):
    val = f.read(0x4)
    return unpack(">f", val)[0]


def read_uint16(f):
    return unpack(">H", f.read(2))[0]


def write_uint32(f, val):
    f.write(pack(">I", val))


def write_uint16(f, val):
    f.write(pack(">H", val))


def write_vector3(f, vector):
    f.write(pack(">fff", vector.x, vector.y, vector.z))


def write_float(f, val):
    f.write(pack(">f", val))


def read_obj(objfile):
    vertices = []
    faces = []
    face_normals = []
    normals = []

    floor_type = None

    smallest_x = smallest_z = biggest_x = biggest_z = None

    for line in objfile:
        line = line.strip()
        args = line.split(" ")

        if len(args) == 0 or line.startswith("#"):
            continue
        cmd = args[0]

        if cmd == "v":
            # print(args)
            for i in range(args.count("")):
                args.remove("")

            x, y, z = map(float, args[1:4])
            vertices.append((x, y, z))

            if smallest_x is None:
                # Initialize values
                smallest_x = biggest_x = x
                smallest_z = biggest_z = z
            else:
                if x < smallest_x:
                    smallest_x = x
                elif x > biggest_x:
                    biggest_x = x
                if z < smallest_z:
                    smallest_z = z
                elif z > biggest_z:
                    biggest_z = z

        elif cmd == "f":
            # if it uses more than 3 vertices to describe a face then we panic!
            # no triangulation yet.
            if len(args) == 5:
                # raise RuntimeError("Model needs to be triangulated! Only faces with 3 vertices are supported.")
                v1, v2, v3, v4 = map(read_vertex, args[1:5])
                # faces.append(((v1[0] - 1, v1[1]), (v3[0] - 1, v3[1]), (v2[0] - 1, v2[1])))
                # faces.append(((v3[0] - 1, v3[1]), (v1[0] - 1, v1[1]), (v4[0] - 1, v4[1])))
                faces.append((v1, v2, v3, floor_type))
                faces.append((v3, v4, v1, floor_type))
            elif len(args) == 4:
                v1, v2, v3 = map(read_vertex, args[1:4])
                # faces.append(((v1[0]-1, v1[1]), (v3[0]-1, v3[1]), (v2[0]-1, v2[1])))
                faces.append((v1, v2, v3, floor_type))
            else:
                raise RuntimeError("Model needs to be triangulated! Only faces with 3 or 4 vertices are supported.")
            # if len(args) != 4:
            #    raise RuntimeError("Model needs to be triangulated! Only faces with 3 vertices are supported.")
            # v1, v2, v3 = map(read_vertex, args[1:4])

            # faces.append((v1, v2, v3, floor_type))

        elif cmd == "vn":
            nx, ny, nz = map(float, args[1:4])
            normals.append((nx, ny, nz))

        elif cmd == "usemtl":
            assert len(args) >= 2

            matname = " ".join(args[1:])

            floor_type_match = match("^(.*?)(0x[0-9a-fA-F]{4})(.*?)$", matname)

            if floor_type_match is not None:
                floor_type = int(floor_type_match.group(2), 16)
            else:
                floor_type = None

            # print("Found material:", matname, "Using floor type:", hex(floor_type))

    # objects.append((current_object, vertices, faces))
    return vertices, faces, normals, (smallest_x, smallest_z, biggest_x, biggest_z)


class BoundaryBox(object):
    def __init__(self):
        self.start = None
        self.end = None
        self.mid = None

    @classmethod
    def from_vector(cls, start, end):
        bbox = cls()
        bbox.start = start.copy()
        bbox.end = end.copy()
        bbox.mid = (bbox.start + bbox.end) / 2.0
        return bbox

    @classmethod
    def from_file(cls, f):
        bbox = cls()
        bbox.start = Vector3(*read_float_tripple(f))
        bbox.end = Vector3(*read_float_tripple(f))
        return bbox

    def write(self, f):
        write_vector3(f, self.start)
        write_vector3(f, self.end)

    def size(self):
        diff = self.end - self.start
        diff.x = abs(diff.x)
        diff.y = abs(diff.y)
        diff.z = abs(diff.z)
        return diff

    def scale(self, x, y, z):
        mid = (self.start + self.end) / 2.0
        p1 = self.start - mid
        p2 = self.end - mid

        p1.x *= x
        p1.y *= y
        p1.z *= z

        p2.x *= x
        p2.y *= y
        p2.z *= z

        self.start = p1 + mid
        self.end = p2 + mid

    def contains(self, triangle):
        p1, p2, p3 = triangle.origin, triangle.p2, triangle.p3
        start, end = self.start, self.end

        min_x = min(p1.x, p2.x, p3.x)# - self.mid.x
        max_x = max(p1.x, p2.x, p3.x)# - self.mid.x

        min_z = min(p1.z, p2.z, p3.z)# - self.mid.z
        max_z = max(p1.z, p2.z, p3.z)# - self.mid.z

        if max_x < start.x or min_x > end.x:
            return False
        if max_z < start.z or min_z > end.z:
            return False

        return True


class BJMPTriangle(object):
    def __init__(self):
        self._p1_index = None
        self._p2_index = None
        self._p3_index = None

        self.triangle = None
        self.data = None
        self.normal = None
        self.d = 0
        self.binormal = None
        self.tangent = None

        self.p1 = None
        self.edge_normal1 = None
        self.edge_normal1_d = 0

        self.p2 = None
        self.edge_normal2 = None
        self.edge_normal2_d = 0

        self.p3 = None
        self.edge_normal3 = None
        self.edge_normal3_d = 0

        self.coll_data = 0x100
    
    def is_wall(self, normal):
        return degrees(acos(normal.cos_angle(Vector3(0.0, 1.0, 0.0)))) > 45

    
    @classmethod
    def from_triangle(cls, triangle, coll_data=None):
        tri = cls()
        tri.triangle = triangle
        triangle.normal *= -1
        round_vector(triangle.normal, 6)

        
        tri.coll_data = coll_data
        tri.normal = triangle.normal

        if not tri.is_wall(tri.normal):
            tri.binormal = triangle.normal.cross(FWVECTOR) #*-1
            flip = True 
            
            if tri.binormal.norm() == 0:
                #tri.binormal = triangle.normal.cross(UPVECTOR) *-1
                #flip = True 
                tri.binormal = UPVECTOR.copy()
            tri.binormal.normalize()
            tri.tangent = Vector3(0.0, 0.0, 0.0)
            tri.tangent = triangle.normal.cross(tri.binormal)#*-1
            tri.tangent.normalize()
            tri.normal = triangle.normal
            if coll_data is None:
                tri.coll_data = 0x0100
        else:
            tri.binormal = triangle.normal.cross(UPVECTOR) #*-1
            flip = True 
            
            if tri.binormal.norm() == 0:
                #tri.binormal = triangle.normal.cross(UPVECTOR) *-1
                #flip = True 
                tri.binormal = FWVECTOR.copy()
            tri.binormal.normalize()
            tri.tangent = Vector3(0.0, 0.0, 0.0)
            tri.tangent = triangle.normal.cross(tri.binormal)#*-1
            tri.tangent.normalize()
            tri.binormal *= -1 
            tri.tangent *= -1
            if coll_data is None:
                tri.coll_data = 0x810
        
        #if flip:
        #    tmp = tri.tangent 
        #    tri.tangent = tri.binormal*-1 
        #    tri.binormal = tmp#*-1
        
        tri.d = tri.normal.dot(triangle.origin)

        p1, p2, p3 = triangle.origin, triangle.p2, triangle.p3
        #tri.p1 = Vector3(-p1.z, 0, -p1.x)
        tri.edge_normal1 = (p2-p1).cross(tri.normal)
        tri.edge_normal1.normalize()
        

        #tri.p2 = Vector3(-p2.z, 0, -p2.x)
        tri.edge_normal2 = (p3-p2).cross(tri.normal)
        tri.edge_normal2.normalize()
        tri.edge_normal2_d = tri.edge_normal2.dot(p2)

        #tri.p3 = Vector3(-p3.z, 0, -p3.x)
        tri.edge_normal3 = (p1-p3).cross(tri.normal)
        tri.edge_normal3.normalize()
        tri.edge_normal3_d = tri.edge_normal3.dot(p3)
        tri.edge_normal1_d = tri.edge_normal1.dot(p1)
        
        nbt = Matrix3x3(
                    tri.binormal.x, tri.binormal.y, tri.binormal.z,
                    tri.normal.x, tri.normal.y, tri.normal.z,
                    tri.tangent.x, tri.tangent.y, tri.tangent.z
                )
        p1 = p1 - tri.normal*tri.d
        p2 = p2 - tri.normal*tri.d
        p3 = p3 - tri.normal*tri.d
        p1 = nbt.multiply_vec3(p1)
        p2 = nbt.multiply_vec3(p2)
        p3 = nbt.multiply_vec3(p3)
        tri.p1 = p1 
        tri.p2 = p2 
        tri.p3 = p3
        """nbt = Matrix3x3(tri.normal.x, tri.normal.y, tri.normal.z,  
        tri.tangent.x, tri.tangent.y, tri.tangent.z,
        tri.binormal.x, tri.binormal.y, tri.binormal.z)
        
        nbt.transpose()
        tri.p1 = Vector3(*nbt.multiply_vec3(p1.x, p1.y, p1.z))
        tri.p2 = Vector3(*nbt.multiply_vec3(p2.x, p2.y, p2.z))
        tri.p3 = Vector3(*nbt.multiply_vec3(p3.x, p3.y, p3.z))"""
        
        return tri

    @classmethod
    def from_file(cls, f, vertices):
        tri = cls()
        start = f.tell()
        v1, v2, v3 = read_uint16(f), read_uint16(f), read_uint16(f)
        tri.triangle = Triangle(vertices[v1], vertices[v2], vertices[v3])
        tri.normal = Vector3(*read_float_tripple(f))


        tri.d = read_float(f)
        tri.binormal = Vector3(*read_float_tripple(f))
        tri.tangent = Vector3(*read_float_tripple(f))

        tri.p1 = Vector3(read_float(f), 0, read_float(f))
        tri.edge_normal1 = Vector3(*read_float_tripple(f))
        tri.edge_normal1_d = read_float(f)

        tri.p2 = Vector3(read_float(f), 0, read_float(f))
        tri.edge_normal2 = Vector3(*read_float_tripple(f))
        tri.edge_normal2_d = read_float(f)

        tri.p3 = Vector3(read_float(f), 0, read_float(f))
        tri.edge_normal3 = Vector3(*read_float_tripple(f))
        tri.edge_normal3_d = read_float(f)
        tri.coll_data = read_uint16(f)
        
        assert f.tell() - start == 0x78
        return tri

    def fill_vertices(self, vertices: list):
        try:
            v1_index = vertices.index(self.triangle.origin)
        except ValueError:
            v1_index = len(vertices)
            vertices.append(self.triangle.origin)

        try:
            v2_index = vertices.index(self.triangle.p2)
        except ValueError:
            v2_index = len(vertices)
            vertices.append(self.triangle.p2)

        try:
            v3_index = vertices.index(self.triangle.p3)
        except ValueError:
            v3_index = len(vertices)
            vertices.append(self.triangle.p3)

        self._p1_index = v1_index
        self._p2_index = v2_index
        self._p3_index = v3_index

    def write(self, f):
        write_uint16(f, self._p1_index)
        write_uint16(f, self._p2_index)
        write_uint16(f, self._p3_index)

        write_vector3(f, self.normal)
        write_float(f, self.d)
        write_vector3(f, self.binormal)
        write_vector3(f, self.tangent)

        write_float(f, self.p1.x)
        write_float(f, self.p1.z)
        write_vector3(f, self.edge_normal1)
        write_float(f, self.edge_normal1_d)

        write_float(f, self.p2.x)
        write_float(f, self.p2.z)
        write_vector3(f, self.edge_normal2)
        write_float(f, self.edge_normal2_d)

        write_float(f, self.p3.x)
        write_float(f, self.p3.z)
        write_vector3(f, self.edge_normal3)
        write_float(f, self.edge_normal3_d)

        write_uint16(f, self.coll_data)


class Group(object):
    def __init__(self):
        self._tri_count = 0
        self._offset = 0
        self.bbox = None
        self.tri_indices = []

    @classmethod
    def from_file(cls, f):
        group = cls()
        val = read_uint32(f)
        group._tri_count = (val >> 24) & 0xFF
        group._offset = val & 0xFFFFFF
        group.bbox = BoundaryBox.from_file(f)

        return group

    def read_indices(self, indices):
        for i in range(self._tri_count):
            self.tri_indices.append(indices[i+self._offset])
    
    def add_indices(self, indices):
        self._offset = len(indices)
        self._tri_count = len(self.tri_indices)
        
        for index in self.tri_indices:
            indices.append(index)
        
    
    def write(self, f):
        assert self._tri_count <= 0xFF
        assert self._offset <= 0xFFFFFF
        write_uint32(f, self._tri_count << 24 | self._offset)
        self.bbox.write(f)


class CollisionGroups(object):
    def __init__(self):
        self.bbox = None
        self.grid_x = 0
        self.grid_y = 0
        self.grid_z = 0
        self.cell_dimensions = None
        self.cell_inverse = None 
        self.groups = []
        #self.indices = []

    @classmethod
    def from_model(cls, model):
        pass

    @classmethod
    def from_file(cls, f):
        colgroups = cls()
        colgroups.bbox = BoundaryBox.from_file(f)
        colgroups.grid_x = read_uint32(f)
        colgroups.grid_y = read_uint32(f)
        colgroups.grid_z = read_uint32(f)
        colgroups.cell_dimensions = read_vector3(f)
        colgroups.cell_inverse = read_vector3(f)
        group_count = read_uint32(f)
        colgroups.groups = []

        for i in range(group_count):
            colgroups.groups.append(Group.from_file(f))

        indices = []
        index_count = read_uint32(f)
        for i in range(index_count):
            indices.append(read_uint16(f))

        for group in colgroups.groups:
            group.read_indices(indices)

        return colgroups

    def write(self, f):
        self.bbox.write(f)
        write_uint32(f, self.grid_x)
        write_uint32(f, self.grid_y)
        write_uint32(f, self.grid_z)
        write_vector3(f, self.cell_dimensions)
        write_vector3(f, self.cell_inverse)
        write_uint32(f, len(self.groups))
        
        indices = []
        for group in self.groups:
            group.add_indices(indices)
            group.write(f)

        indices = []
        for group in self.groups:
            indices.extend(group.tri_indices)

        write_uint32(f, len(indices))
        for index in indices:
            write_uint16(f, index)


class BJMP(object):
    def __init__(self):
        self.bbox_inner = None
        self.bbox_outer = None
        self.triangles = []
        self.collision_groups = CollisionGroups()

    @classmethod
    def from_obj(cls, f):
        vertices = []
        uvs = []
        faces = []
        bjmp = cls()
        collision_type = None


        smallest_x = smallest_y = smallest_z = biggest_x = biggest_y = biggest_z = None

        for line in f:
            line = line.strip()
            args = line.split(" ")

            if len(args) == 0 or line.startswith("#"):
                continue
            cmd = args[0]

            if cmd == "v":
                # print(args)
                for i in range(args.count("")):
                    args.remove("")

                x, y, z = map(float, args[1:4])
                vertices.append(Vector3(x, y, z))

                if smallest_x is None:
                    # Initialize values
                    smallest_x = biggest_x = x
                    smallest_y = biggest_y = y
                    smallest_z = biggest_z = z
                else:
                    if x < smallest_x:
                        smallest_x = x
                    elif x > biggest_x:
                        biggest_x = x

                    if y < smallest_y:
                        smallest_y = y
                    elif y > biggest_y:
                        biggest_y = y

                    if z < smallest_z:
                        smallest_z = z
                    elif z > biggest_z:
                        biggest_z = z

            elif cmd == "f":
                # if it uses more than 3 vertices to describe a face then we panic!
                # no triangulation yet.
                if len(args) == 5:
                    # raise RuntimeError("Model needs to be triangulated! Only faces with 3 vertices are supported.")
                    v1, v2, v3, v4 = map(read_vertex, args[1:5])
                    # faces.append(((v1[0] - 1, v1[1]), (v3[0] - 1, v3[1]), (v2[0] - 1, v2[1])))
                    # faces.append(((v3[0] - 1, v3[1]), (v1[0] - 1, v1[1]), (v4[0] - 1, v4[1])))
                    tri1 = Triangle(vertices[v1 - 1], vertices[v3 - 1], vertices[v2 - 1])
                    tri2 = Triangle(vertices[v3 - 1], vertices[v1 - 1], vertices[v4 - 1])
                    
                    
                    
                    if tri1.normal.norm() != 0:
                        bjmp_tri1 = BJMPTriangle.from_triangle(tri1, collision_type)
                        bjmp.triangles.append(bjmp_tri1)
                    if tri2.normal.norm() != 0:
                        bjmp_tri2 = BJMPTriangle.from_triangle(tri2, collision_type)
                        bjmp.triangles.append(bjmp_tri2)
                elif len(args) == 4:
                    v1, v3, v2 = map(read_vertex, args[1:4])
                    # faces.append(((v1[0]-1, v1[1]), (v3[0]-1, v3[1]), (v2[0]-1, v2[1])))
                    tri1 = Triangle(vertices[v1 - 1], vertices[v2 - 1], vertices[v3 - 1])
                    
                    if tri1.normal.norm() != 0:
                        bjmp_tri1 = BJMPTriangle.from_triangle(tri1, collision_type)
                        bjmp.triangles.append(bjmp_tri1)
                else:
                    raise RuntimeError(
                        "Model needs to be triangulated! Only faces with 3 or 4 vertices are supported.")
                # if len(args) != 4:
                #    raise RuntimeError("Model needs to be triangulated! Only faces with 3 vertices are supported.")
                # v1, v2, v3 = map(read_vertex, args[1:4])

                # faces.append((v1, v2, v3, floor_type))
            elif cmd == "usemtl":
                assert len(args) >= 2

                matname = " ".join(args[1:])

                floor_type_match = match("^(.*?)(0x[0-9a-fA-F]{4})(.*?)$", matname)

                if floor_type_match is not None:
                    collision_type = int(floor_type_match.group(2), 16)
                else:
                    collision_type = None

                # print("Found material:", matname, "Using floor type:", hex(floor_type))"""

        bjmp.bbox_inner = BoundaryBox.from_vector(
            Vector3(smallest_x, smallest_y, smallest_z),
            Vector3(biggest_x, biggest_y, biggest_z)
        )

        bjmp.bbox_outer = BoundaryBox.from_vector(
            bjmp.bbox_inner.start,
            bjmp.bbox_inner.end
        )

        cell_x = 150.0
        cell_z = 150.0

        bjmp.collision_groups.bbox = bjmp.bbox_inner
        bjmp.collision_groups.cell_dimensions = Vector3(cell_x, biggest_y - smallest_y, cell_z)
        bjmp.collision_groups.cell_inverse = Vector3(  1.0/bjmp.collision_groups.cell_dimensions.x,
                                                    1.0/bjmp.collision_groups.cell_dimensions.y, 
                                                    1.0/bjmp.collision_groups.cell_dimensions.z)

        x_max = int(ceil((biggest_x - smallest_x) / cell_x))
        z_max = int(ceil((biggest_z - smallest_z) / cell_z))
        
        start_x = bjmp.bbox_inner.start.x
        start_z = bjmp.bbox_inner.start.z
        
        bjmp.collision_groups.grid_x = x_max
        bjmp.collision_groups.grid_y = 1
        bjmp.collision_groups.grid_z = z_max
        
        for ix in range(x_max):
            print(ix, "/", x_max)
            for iz in range(z_max):
                bbox_x = start_x + ix*cell_x 
                bbox_z = start_z + iz*cell_z 
                
                bbox = BoundaryBox.from_vector(
                    Vector3(bbox_x, smallest_y, bbox_z),
                    Vector3(bbox_x+cell_x, biggest_y, bbox_z+cell_z)
                )
                
                group = Group()
                group.bbox = bbox 
                min_y = inf
                max_y = -inf
                for i, triangle in enumerate(bjmp.triangles):
                    if bbox.contains(triangle.triangle):
                        tri = triangle.triangle 
                        if tri.origin.y < min_y:
                            min_y = tri.origin.y
                        if tri.p2.y < min_y:
                            min_y = tri.p2.y 
                        if tri.p3.y < min_y:
                            min_y = tri.p3.y 
                        
                        if tri.origin.y > max_y:
                            max_y = tri.origin.y
                        if tri.p2.y > max_y:
                            max_y = tri.p2.y 
                        if tri.p3.y > max_y:
                            max_y = tri.p3.y 
                        
                        group.tri_indices.append(i)
                if min_y < bbox.start.y:
                    bbox.start.y = min_y
                if max_y > bbox.start.y:
                    bbox.end.y = max_y
                bbox.start.y -= 5.0
                bbox.end.y += 5.0
                bjmp.collision_groups.groups.append(group)
        
        
                        


        return bjmp

    @classmethod
    def from_file(cls, f):
        bjmp = cls()

        magic = read_uint32(f)
        if magic == 0x013304E6:
            #self.simple = False
            bjmp.bbox_inner = BoundaryBox.from_file(f)
            bjmp.bbox_outer = BoundaryBox.from_file(f)
        #elif magic == 0x01330237:
        #    self.simple = True
        #    self.bbox = BoundaryBox()
        else:
            raise RuntimeError("Unknown/Unsupported magic: {:x}".format(magic))

        vertex_count = read_uint16(f)

        vertices = []
        for i in range(vertex_count):
            vertices.append(read_vector3(f))

        bjmp.triangles = []
        tri_count = read_uint32(f)
        for i in range(tri_count):
            bjmp.triangles.append(BJMPTriangle.from_file(f, vertices))
        print("Remaining data starts at {0:x}".format(f.tell()))
        bjmp.collision_groups = CollisionGroups.from_file(f)
        assert f.read() == b""
        print("sizes")
        print("x z size:", bjmp.collision_groups.grid_x, bjmp.collision_groups.grid_z)
        print(bjmp.collision_groups.bbox.size())
        print(bjmp.collision_groups.cell_dimensions)

        return bjmp

    def write(self, f):
        write_uint32(f, 0x013304E6)
        self.bbox_inner.write(f)
        self.bbox_outer.write(f)

        vertices = []
        for triangle in self.triangles:
            triangle.fill_vertices(vertices)

        write_uint16(f, len(vertices))
        for vertex in vertices:
            write_vector3(f, vertex)

        write_uint32(f, len(self.triangles))
        for triangle in self.triangles:
            triangle.write(f)

        self.collision_groups.write(f)
        
if __name__ == "__main__":
    import sys 
    in_name = sys.argv[1]
    if in_name.endswith(".obj"):
        out_name = in_name + ".bjmp"
        with open(in_name, "r") as f:
            bjmp = BJMP.from_obj(f)
        with open(out_name, "wb") as f:
            bjmp.write(f)
    elif in_name.endswith(".bjmp"):
        out_name = in_name+".obj"
        with open(in_name, "rb") as f:
            bjmp = BJMP.from_file(f)
        
        with open(out_name, "w") as f:
            f.write("# .OBJ generated from Pikmin 2 by Yoshi2's obj2grid.py\n\n")
            f.write("# VERTICES BELOW\n\n") 
            vertex_counter = 0
            faces = []
            for btriangle in bjmp.triangles:
                tri = btriangle.triangle 
                p1, p2, p3 = tri.origin, tri.p2, tri.p3 
                
                f.write("v {} {} {}\n".format(p1.x, p1.y, p1.z))
                f.write("v {} {} {}\n".format(p2.x, p2.y, p2.z))
                f.write("v {} {} {}\n".format(p3.x, p3.y, p3.z))
                #f.write("vt {} {}\n".format(btriangle.p1.x, btriangle.p1.z))
                #f.write("vt {} {}\n".format(btriangle.p2.x, btriangle.p2.z))
                #f.write("vt {} {}\n".format(btriangle.p3.x, btriangle.p3.z))
                
                faces.append((vertex_counter+1, vertex_counter+2, vertex_counter+3, btriangle.coll_data))
                vertex_counter += 3 
            
            last_coll = None 
            
            for i1, i2, i3, coll in faces:
                if coll != last_coll:
                    f.write("usemtl collision_type0x{:04X}\n".format(coll))
                f.write("f {0} {2} {1}\n".format(i1, i2, i3))
    print("done")