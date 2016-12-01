#define RGB2Y(R, G, B) (( (  66 * (R) + 129 * (G) +  25 * (B) + 128) >> 8) +  16)
#define RGB2U(R, G, B) (( ( -38 * (R) -  74 * (G) + 112 * (B) + 128) >> 8) + 128)
#define RGB2V(R, G, B) (( ( 112 * (R) -  94 * (G) -  18 * (B) + 128) >> 8) + 128)

__kernel	void xrgb_to_i420_kernel(
		__global uchar4 *input_img,
		__global ulong *Y,
		__global uint *U,
		__global uint *V,
		int rgbstride, //size in bytes/4
		int ystride, //size in bytes / 2
		int uvstride) //size in bytes /2
{
	const int img_width = get_global_size(0);
	const int col = get_global_id(0)<<2;
	const int quad = get_global_id(0);
	const int row = get_global_id(1);
	const int hrow = get_global_id(1)>>1;

	uchar4 xrgb;
	ulong y = 0, i, aux;
	uint u=0,v=0;

	for ( i=0; i<8; i++ ) {
		xrgb = input_img[row*rgbstride + col +i];
		aux = (RGB2Y(xrgb.y, xrgb.z, xrgb.w));
		y |= ((uchar) aux) << (8 * i);
		if (i&1 && row&1){
			u |= (RGB2U(xrgb.y, xrgb.z, xrgb.w))<<(8*(i>>1));
			v |= (RGB2V(xrgb.y, xrgb.z, xrgb.w))<<(8*(i>>1));
		}
	}
	Y[quad + row*ystride] = y;
	U[quad + hrow*uvstride] = u;
	V[quad + hrow*uvstride] = v;

};
