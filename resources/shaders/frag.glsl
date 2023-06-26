#version 120

// uniform sampler2D aoTexture;
// uniform vec2 windowSize;
uniform vec3 lightPos;
uniform vec3 ka;
uniform vec3 kd;
uniform vec3 ks;
uniform float s;

varying vec3 vPos; // vertex position
varying vec3 vNor; // vertex normal

void main()
{
	// vec2 tex;
    // tex.x = gl_FragCoord.x/windowSize.x;
    // tex.y = gl_FragCoord.y/windowSize.y;
    
    // // Fetch shading data
    // vec3 aoFactor = texture2D(aoTexture, vTex);
    
    // // Indirect Lighting Approximation (Ambient Occlusion)
	vec3 l_hat = normalize(lightPos - vPos);
	vec3 e_hat = normalize(-vPos);
	vec3 h_hat = normalize(l_hat + e_hat);
	vec3 n_hat = normalize(vNor);

	vec3 ambient = ka;
	vec3 diffuse = kd * max(0, dot(l_hat, n_hat));
	vec3 specular = ks * pow(max(0, dot(h_hat, n_hat)), s);
	vec3 color = ambient + diffuse + specular;
	gl_FragColor = vec4(color, 1.0);
}
