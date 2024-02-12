# GLSL - 3D Sphere Rendering with Shaders

## Project Description:

This project, developed within the field of Computer Graphics and Image Processing, was created as part of a lecture and exercises. It utilizes the Python programming language and the libraries Pygame, NumPy, Pillow (PIL), and OpenGL. The project's goal is to display and rotate a textured 3D sphere in an interactive window.

## Implemented Features:

1. **Shader Programming:** The project utilizes Vertex and Fragment shaders for the representation and manipulation of 3D objects.

2. **Texturing:** A texture (in this case, "test.jpeg") is mapped onto the 3D sphere to create a realistic surface.

3. **3D Transformations:** Matrices are used to apply 3D transformations (rotation) to the model.

4. **Shader Filter:** There is an option to apply a grayscale filter to the texture. The variable `use_filter` controls this filter.

5. **Pygame Integration:** Pygame is used to create an interactive window and enable the rotation of the sphere. The window is rendered in OpenGL.

## Shaders:

The shaders are written in GLSL and facilitate the lighting of the 3D object. The Fragment shader also contains a condition (`if (uUseFilter)`) for the optional grayscale filter.

## Code Structure:

- The function `load_program` creates a shader program and links Vertex and Fragment shaders.
- `load_shader` loads a single shader and compiles it.
- `load_texture` loads a texture from a file and binds it to OpenGL.
- `perspective` creates a perspective projection matrix.
- `rotate` performs a rotation operation on the model.
- `generate_sphere` creates the vertices, normals, and texture coordinates for a sphere.

The main loop (`while running`) enables the rotation of the sphere and responds to window closure (`pygame.QUIT`).

In summary, the project provides a practical implementation of fundamental concepts in computer graphics and OpenGL programming.
