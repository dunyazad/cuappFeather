#version 330 core

in vec3 vNormal;
in vec4 vColor;
in vec2 vUV;
in vec3 vFragPos;

uniform vec3 cameraPos;
uniform sampler2D texture0; // ���� �߰�: �ؽ�ó ������

out vec4 FragColor;

void main() {
    vec3 lightDir = normalize(cameraPos - vFragPos);
    float lighting = max(dot(normalize(vNormal), lightDir), 0.2);

//    vec4 texColor = texture(texture0, vUV); // UV�κ��� �ؽ�ó �� ��������
//    vec4 finalColor = vColor * texColor;    // vertex color�� texture color ���ϱ�
//    FragColor = finalColor * lighting;      // ���� ����

    FragColor = texture(texture0, vUV); // UV�κ��� �ؽ�ó �� ��������
}
