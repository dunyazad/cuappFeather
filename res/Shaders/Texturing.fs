#version 330 core

in vec3 vNormal;
in vec4 vColor;
in vec2 vUV;
in vec3 vFragPos;

uniform vec3 cameraPos;
uniform sampler2D texture0; // 새로 추가: 텍스처 유니폼

out vec4 FragColor;

void main() {
    vec3 lightDir = normalize(cameraPos - vFragPos);
    float lighting = max(dot(normalize(vNormal), lightDir), 0.2);

//    vec4 texColor = texture(texture0, vUV); // UV로부터 텍스처 색 가져오기
//    vec4 finalColor = vColor * texColor;    // vertex color와 texture color 곱하기
//    FragColor = finalColor * lighting;      // 조명 적용

    FragColor = texture(texture0, vUV); // UV로부터 텍스처 색 가져오기
}
