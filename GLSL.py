# Abgabe von Valentino Giorgio Pino, Matr.Nr.: 2225371

# Die Funktionen wurden mithilfe von den Bibliotheken von Python, OpenGL geschrieben, außerdem wurde das Wissen
# aus der Vorlesung/Übung angewendet und schlussendlich für einige Hilfestellungen ChatGPT/Phind genutzt

# Initialisierung der benötigten Bibliotheken und Module
from __future__ import division
from OpenGL.GL import *
import numpy as np
import math
import pygame
import textwrap
from PIL import Image

use_filter = False  # Variable zur Steuerung des Filters

# Shader-Quellcode für den Vertex-Shader
vertex_shader_source = textwrap.dedent("""\
    uniform mat4 uMVMatrix;
    uniform mat4 uPMatrix;

    attribute vec3 aVertex;
    attribute vec3 aNormal;
    attribute vec2 aTexCoord;

    varying vec2 vTexCoord;

    void main(){
       vTexCoord = aTexCoord;
       // Make GL think we are actually using the normal
       aNormal;
       gl_Position = (uPMatrix * uMVMatrix)  * vec4(aVertex, 1.0);
    }
    """)

# Shader-Quellcode für den Fragment-Shader
fragment_shader_source = textwrap.dedent("""\
    uniform sampler2D sTexture;
    uniform bool uUseFilter;

    varying vec2 vTexCoord;

    void main(){
       vec4 color = texture2D(sTexture, vTexCoord);

       if (uUseFilter) {
           float grayscale = (color.r + color.g + color.b) / 3.0;
           color = vec4(grayscale, grayscale, grayscale, color.a);
       }

       gl_FragColor = color;
    }
    """)


# Funktion zum Laden eines Shader-Programms
def load_program(vertex_source, fragment_source):
    # Laden des Vertex-Shaders
    vertex_shader = load_shader(GL_VERTEX_SHADER, vertex_source)
    if vertex_shader == 0:
        return 0

    # Laden des Fragment-Shaders
    fragment_shader = load_shader(GL_FRAGMENT_SHADER, fragment_source)
    if fragment_shader == 0:
        return 0

    # Erzeugung des Shader-Programms
    program = glCreateProgram()

    if program == 0:
        return 0

    # Anfügen der Shader an das Programm
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)

    # Verknüpfen und Kompilieren des Shader-Programms
    glLinkProgram(program)

    if glGetProgramiv(program, GL_LINK_STATUS, None) == GL_FALSE:
        glDeleteProgram(program)
        return 0

    # Verwendung des Shader-Programms
    glUseProgram(program)

    # Abrufen der Uniform- und Attribut-Positionen im Shader-Programm
    uMVMatrix = glGetUniformLocation(program, "uMVMatrix")
    uPMatrix = glGetUniformLocation(program, "uPMatrix")
    sTexture = glGetUniformLocation(program, "sTexture")
    uUseFilter = glGetUniformLocation(program, "uUseFilter")  # Hinzugefügte Zeile für den Filterzustand

    return program


# Funktion zum Laden eines Shaders
def load_shader(shader_type, source):
    shader = glCreateShader(shader_type)

    if shader == 0:
        return 0

    # Setzen des Shader-Quellcodes
    glShaderSource(shader, source)
    glCompileShader(shader)

    if glGetShaderiv(shader, GL_COMPILE_STATUS, None) == GL_FALSE:
        info_log = glGetShaderInfoLog(shader)
        print(info_log)
        glDeleteProgram(shader)
        return 0

    return shader


# Laden einer Textur aus einer Datei
def load_texture(filename):
    img = Image.open(filename, 'r').convert("RGB")
    img_data = np.array(img, dtype=np.uint8)
    w, h = img.size

    texture = glGenTextures(1)

    # Binden der Textur und Einstellung der Filterparameter
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

    # Übertragen der Texturdaten auf den Grafikspeicher
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)

    return texture


# Funktion zur Erzeugung einer Perspektivischen Projektionsmatrix
def perspective(fovy, aspect, z_near, z_far):
    f = 1 / math.tan(math.radians(fovy) / 2)
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (z_far + z_near) / (z_near - z_far), -1],
        [0, 0, (2 * z_far * z_near) / (z_near - z_far), 0]
    ])


# Funktion zur Rotation eines Objekts um eine Achse
def rotate(angle, x, y, z):
    s = math.sin(math.radians(angle))
    c = math.cos(math.radians(angle))
    magnitude = math.sqrt(x * x + y * y + z * z)
    nc = 1 - c

    x /= magnitude
    y /= magnitude
    z /= magnitude

    return np.array([
        [c + x ** 2 * nc, y * x * nc - z * s, z * x * nc + y * s, 0],
        [y * x * nc + z * s, c + y ** 2 * nc, y * z * nc - x * s, 0],
        [z * x * nc - y * s, z * y * nc + x * s, c + z ** 2 * nc, 0],
        [0, 0, 0, 1],
    ])


# Funktion zur Erzeugung der Vertices, Normals und Texcoords einer Kugel
def generate_sphere(radius, stacks, sectors):
    vertices = []
    normals = []
    texcoords = []

    # Erzeugung der Vertices, Normals und Texcoords
    for stack in range(stacks + 1):
        theta = stack * math.pi / stacks
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)

        for sector in range(sectors + 1):
            phi = sector * 2 * math.pi / sectors
            sin_phi = math.sin(phi)
            cos_phi = math.cos(phi)

            x = radius * cos_phi * sin_theta
            y = radius * sin_phi * sin_theta
            z = radius * cos_theta

            vertices.append((x, y, z))
            normals.append((x, y, z))
            texcoords.append((phi / (2 * math.pi), theta / math.pi))

    return vertices, normals, texcoords


radius = 1.0
stacks = 20
sectors = 20

# Erzeugung der Vertices, Normals und Texcoords einer Kugel
_vertices, _normals, _texcoords = generate_sphere(radius, stacks, sectors)

# Erzeugung der Vertex-Triangles, Normal-Triangles und Texture-Triangles
_vertex_triangles = []
_normal_triangles = []
_texture_triangles = []

for stack in range(stacks):
    for sector in range(sectors):
        v1 = stack * (sectors + 1) + sector
        v2 = v1 + 1
        v3 = (stack + 1) * (sectors + 1) + sector
        v4 = v3 + 1

        _vertex_triangles.append((v1, v2, v3))
        _vertex_triangles.append((v2, v4, v3))

        _normal_triangles.append((v1, v2, v3))
        _normal_triangles.append((v2, v4, v3))

        _texture_triangles.append((v1, v2, v3))
        _texture_triangles.append((v2, v4, v3))

# Erzeugung von numpy-Arrays für die Vertices, Normals und Texcoords
_vertices = np.array([
    _vertices[index]
    for indices in _vertex_triangles
    for index in indices
])

_normals = np.array([
    _normals[index]
    for indices in _normal_triangles
    for index in indices
])

_texcoords = np.array([
    _texcoords[index]
    for indices in _texture_triangles
    for index in indices
])

if __name__ == "__main__":
    width, height = 800, 600
    pygame.display.set_mode((width, height), pygame.DOUBLEBUF | pygame.OPENGL | pygame.HWSURFACE)

    glViewport(0, 0, width, height)
    projection_matrix = perspective(45, width / height, 0.1, 500)
    model_matrix = np.identity(4, dtype=np.float32)
    view_matrix = np.identity(4, dtype=np.float32)

    view_matrix[-1, :-1] = (0, 0, -10)

    program = load_program(vertex_shader_source, fragment_shader_source)

    uMVMatrix = glGetUniformLocation(program, "uMVMatrix")
    uPMatrix = glGetUniformLocation(program, "uPMatrix")
    sTexture = glGetUniformLocation(program, "sTexture")
    uUseFilter = glGetUniformLocation(program, "uUseFilter")

    aVertex = glGetAttribLocation(program, "aVertex")
    aNormal = glGetAttribLocation(program, "aNormal")
    aTexCoord = glGetAttribLocation(program, "aTexCoord")

    glUseProgram(program)
    glEnableVertexAttribArray(aVertex)
    glEnableVertexAttribArray(aNormal)
    glEnableVertexAttribArray(aTexCoord)

    texture = load_texture("test.jpeg")

    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, texture)
    glUniform1i(uUseFilter, int(use_filter))

    glEnable(GL_DEPTH_TEST)

    pygame.init()
    clock = pygame.time.Clock()
    last_tick = pygame.time.get_ticks()

    # Hauptschleife der Anwendung
    running = True
    while running:
        current_tick = pygame.time.get_ticks()
        elapsed_time = (current_tick - last_tick) / 1000.0
        last_tick = current_tick

        angle = elapsed_time * 90
        rotation_matrix = rotate(angle, 0, 1, 0)
        model_matrix = np.dot(rotation_matrix, model_matrix)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glVertexAttribPointer(aVertex, 3, GL_FLOAT, GL_FALSE, 0, _vertices)
        glVertexAttribPointer(aNormal, 3, GL_FLOAT, GL_FALSE, 0, _normals)
        glVertexAttribPointer(aTexCoord, 2, GL_FLOAT, GL_FALSE, 0, _texcoords)

        mv_matrix = np.dot(model_matrix, view_matrix)
        glUniformMatrix4fv(uMVMatrix, 1, GL_FALSE, mv_matrix)
        glUniformMatrix4fv(uPMatrix, 1, GL_FALSE, projection_matrix)

        glDrawArrays(GL_TRIANGLES, 0, len(_vertices))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    use_filter = not use_filter
                    glUniform1i(uUseFilter, int(use_filter))
