#pragma once
#ifndef Texture_H
#define Texture_H

#define GLEW_STATIC
#include <GL/glew.h>

#include <string>

class Texture
{
public:
	Texture();
	virtual ~Texture();
	void setFilename(const std::string &f) { filename = f; }
	void load();
	void init();
	void setUnit(GLint u) { unit = u; }
	GLint getUnit() const { return unit; }
	void bind(GLint handle);
	void unbind();
	void setWrapModes(GLint wrapS, GLint wrapT); // Must be called after init()
	unsigned char getPixel(int r, int c);
	int getWidth() { return width; }
	int getHeight() { return height; }
	
private:
	std::string filename;
	unsigned char *data;
	int width;
	int height;
	int components;
	GLuint tid;
	GLint unit;
	
};

#endif
