#version 330 core

in vec3 vNormal;
in vec4 vColor;
in vec2 vUV;
in vec3 vFragPos; // Fragment position in world space (must be passed from vertex shader)

uniform vec3 cameraPos; // Camera (light) position in world space

out vec4 FragColor;

void main() {
    vec3 lightDir = normalize(cameraPos - vFragPos); // Light direction from fragment to camera
    float lighting = max(dot(normalize(vNormal), lightDir), 0.2); // Diffuse shading with ambient
    FragColor = vColor * lighting; // Apply lighting to color
}
