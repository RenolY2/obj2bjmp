# obj2bjmp
Tool for converting between Pikmin 3 BJMP files and .obj files for import into a 3d modelling program.

Requires a recent version of Python 3 installed. On Windows, make sure to tick the "Add Python to Path" option.

Drag and drop a bjmp or obj file onto convert.bat or libjmp.py to do conversion in the other direction.

bjmp->obj: Collision properties are written as material names with a 4 digit hexadecimal number prefixed with 0x.

obj->bjmp: Collision properties are read from the material names, the name needs to contain a 4 digit hexadecimal number 
prefixed with 0x. If no such number is found, the converter chooses between 0x0100 (ground collision) and 0x0810 (wall collision)
based on angle of triangles.