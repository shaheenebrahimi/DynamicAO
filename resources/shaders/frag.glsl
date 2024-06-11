#version 120

uniform sampler2D aoTexture;
uniform sampler2D genTexture;
uniform vec3 lightPos;
uniform vec3 ka;
uniform vec3 kd;
uniform vec3 ks;
uniform float s;
uniform bool groundTruth;

varying vec3 vPos; // vertex position
varying vec3 vNor; // vertex normal
varying vec2 vTex; // vertex texture coord
varying float vOcc; // vertex occlusion factor

void main()
{
	if (groundTruth) {
		// Fetch shading data
    	vec3 texture = texture2D(aoTexture, vTex).rgb;
		gl_FragColor.rgb = texture;
	}
	else {
		// Blinn Phong
		// vec3 l_hat = normalize(lightPos - vPos);
		// vec3 e_hat = normalize(-vPos);
		// vec3 h_hat = normalize(l_hat + e_hat);
		// vec3 n_hat = normalize(vNor);
		// float ambientFactor = 1.0 - vOcc; // turn to color
		// vec3 ambient = vec3(ambientFactor, ambientFactor, ambientFactor); // ka
		// vec3 diffuse = kd * max(0, dot(l_hat, n_hat));
		// vec3 specular = ks * pow(max(0, dot(h_hat, n_hat)), s);
		// vec3 color = ambient + diffuse + specular; 
		// gl_FragColor.rgb = ambient;
		vec2 flippedTex = vec2(vTex.x, 1.0-vTex.y);
		float ao = texture2D(genTexture, flippedTex).r;
		gl_FragColor.rgb = clamp(vec3(ao,ao,ao), vec3(0.0,0.0,0.0), vec3(1.0f,1.0f,1.0f));
	}
}
