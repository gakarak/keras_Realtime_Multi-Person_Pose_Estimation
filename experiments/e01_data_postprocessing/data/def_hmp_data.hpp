
#ifndef DEF_HMP_DATA
#define DEF_HMP_DATA

struct SimpleData{
	unsigned int rows;
	unsigned int cols;
	unsigned int channels;
	double data[];
};

extern SimpleData HMP_CF;
extern SimpleData HMP_CL;
extern SimpleData PAF_CF;
extern SimpleData PAF_CL;

#endif // DEF_HMP_DATA