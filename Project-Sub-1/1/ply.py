import numpy as np
import os
from pyparsing import col

class Ply(object):
    """Class to represent a ply in memory, read plys, and write plys.
    """

    def __init__(self, ply_path=None, triangles=None, points=None, normals=None, colors=None):
        """Initialize the in memory ply representation.

        Args:
            ply_path (str, optional): Path to .ply file to read (note only
                supports text mode, not binary mode). Defaults to None.
            triangles (numpy.array [k, 3], optional): each row is a list of point indices used to
                render triangles. Defaults to None.
            points (numpy.array [n, 3], optional): each row represents a 3D point. Defaults to None.
            normals (numpy.array [n, 3], optional): each row represents the normal vector for the
                corresponding 3D point. Defaults to None.
            colors (numpy.array [n, 3], optional): each row represents the color of the
                corresponding 3D point. Defaults to None.
        """
        super().__init__()
        # TODO: If ply path is None, load in triangles, point, normals, colors.
        #       else load ply from file.
        #       If ply_path is specified AND other inputs are specified as well, ignore other inputs.
        # TODO: If normals are not None make sure that there are equal number of points and normals.
        # TODO: If colors are not None make sure that there are equal number of colors and normals.

        if ply_path is not None:
            self.read(ply_path)
        else:
            if points is not None:
                if normals is not None and normals.shape[0] != points.shape[0]:
                    raise AssertionError("Number of points and normals must be equal")
                if colors is not None and colors.shape[0] != points.shape[0]:
                    raise AssertionError("Number of points and colors must be equal")

            self.triangles = triangles
            self.points = points
            self.normals = normals
            self.colors = colors

        # pass

    def write(self, ply_path):
        """Write mesh, point cloud, or oriented point cloud to ply file.

        Args:
            ply_path (str): Output ply path.
        """
        # TODO: Write header depending on existance of normals, colors, and triangles.

        if self.points is None:
            raise ValueError("Point is not available")

        PointsCount = self.points.shape[0]##查询点的数量
        Allmatrix = np.copy(self.points)##将点放入矩阵中
        TriangleCount = 0

        if self.normals is not None:##检验向量是否存在（这里不许要报错，因为向量是否存在并非逻辑正确的充分条件）
            if self.normals.shape[0] != PointsCount:
                raise ValueError("Number of normals must match number of points")
            Allmatrix = np.hstack((Allmatrix, self.normals))

        if self.colors is not None:
            if self.colors.shape[0] != PointsCount:
                raise ValueError("Number of colors must match number of points")
            Allmatrix = np.hstack((Allmatrix, self.colors))

        if self.triangles is not None:
            TriangleCount = self.triangles.shape[0]

        with open(ply_path, "w") as file:
            file.write("ply\nformat ascii 1.0\nelement vertex " + str(PointsCount) + "\n")
        # TODO: Write points.
            if self.points is not None:
                file.write("property float x\n")
                file.write("property float y\n")
                file.write("property float z\n")
        # TODO: Write normals if they exist.
            if self.normals is not None:
                if self.normals.shape[0] != PointsCount:
                    raise ValueError("Number of normals must match number of points")
                file.write("property float nx\n")
                file.write("property float ny\n")
                file.write("property float nz\n")
        # TODO: Write colors if they exist.
            if self.colors is not None:
                if self.colors.shape[0] != PointsCount:
                    raise ValueError("Number of colors must match number of points")
                file.write("property uchar red\n")
                file.write("property uchar green\n")
                file.write("property uchar blue\n")
        # TODO: Write face list if needed.
            if self.triangles is not None:
                file.write("element face " + str(TriangleCount) + "\n")
                file.write("property list uchar int vertex_index\n")

            file.write("end_header\n")

            if self.points is not None:
                for i in range(PointsCount):
                    file.write(str(self.points[i][0]) + " " + str(self.points[i][1]) + " " + str(self.points[i][2]))
                    if self.normals is not None:
                        file.write(" " + str(self.normals[i][0]) + " " + str(self.normals[i][1]) + " " + str(self.normals[i][2]))
                    if self.colors is not None:
                        file.write(" " + str(int(self.colors[i][0])) + " " + str(int(self.colors[i][1])) + " " + str(int(self.colors[i][2])))
                    file.write("\n")

            if self.triangles is not None:
                for i in range(TriangleCount):
                    file.write("3 " + str(self.triangles[i][0]) + " " + str(self.triangles[i][1]) + " " + str(self.triangles[i][2]) + "\n")

        file.close()
        pass

    def read(self, ply_path):
        """Read a ply into memory.

        Args:
            ply_path (str): ply to read in.
        """
        # TODO: Read in ply.

        self.colors, self.normals, self.points, self.triangles = [None for _ in range(4)]

        ply_data = []  ##进行初始化和声明
        with open(ply_path, 'r') as file:
            header_ended, NormalExist, ColorlExist = False, False, False
            PointsCount,FaceCount = 0,0

            for line in file:
                if 'element vertex' in line:
                    words = line.split()
                    PointsCount = int(words[2])
                if 'element face' in line:
                    words = line.split()
                    FaceCount = int(words[2])
                if 'nx' in line:
                    NormalExist = True
                if 'red' in line:
                    ColorlExist = True
                if header_ended and line != ' ':  ##文件头部文件已经处理完毕并且当前行不为空时：
                    ply_data.append(line.strip().split(' '))  ##将每一行头部和尾部的空格去掉并用空格分割字符，所有的字符转换为数字格式，并用map进行迭代。然后再将数字转换为整数并用map迭代。最后以列表的形式放入ply_data中
                if line.startswith('end_header'):
                    header_ended = True
        file.close()

        PointsData = np.copy(ply_data[:PointsCount])

        if NormalExist:
            self.points = PointsData[:, :3]
            self.normals = PointsData[:, 3:6]
            if ColorlExist:
                self.colors = PointsData[:, 6:]
        else:
            self.points = PointsData[:, :3]
            if ColorlExist:
                self.colors = PointsData[:, 3:6]

        if FaceCount > 0:
            triangles = np.array(ply_data[PointsCount:])
            self.triangles = triangles[:, 1:].astype(int)


# check = Ply()
# check.read('E:\PythonProgram\HomeWork1\hw1\data\point_sample.ply')
# check.write('E:\PythonProgram\HomeWork1\hw1\data\point.ply')
# check.read('E:\PythonProgram\HomeWork1\hw1\data\\triangle_sample.ply')
# check.write('E:\PythonProgram\HomeWork1\hw1\data\\triangle.ply')
