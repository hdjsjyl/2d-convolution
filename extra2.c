#include <emmintrin.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>
int conv2D(float* in, float* out, int data_size_X, int data_size_Y,
                    float* kernel, int kernel_x, int kernel_y)
{
    omp_set_num_threads(8);
    // the x coordinate of the kernel's center
                int kern_cent_X = (kernel_x - 1)/2;
    // the y coordinate of the kernel's center
                int kern_cent_Y = (kernel_y - 1)/2;

                int x, y, i, j, big;
                float* out_index;
                int small = data_size_X + (2 * kern_cent_X);
                int small2 = data_size_Y+(2*kern_cent_Y);

                __m128 kernel_vector,padded_vector,product_vector, v_a, v_b, v_c, v_d, s_a, s_b, s_c, s_d;
                __m128 sum_vector = _mm_setzero_ps();

    //matrix padding
    float padded[small * small2];

    memset(padded,0,4*small*kern_cent_Y);
    memset(padded+(small*small2)-(small*kern_cent_Y),0,4*small*kern_cent_Y);
    #pragma omp parallel
    {   
       #pragma omp for private(i)
       for(i=0; i<data_size_Y; ++i){
           padded[small*(i+kern_cent_Y)] = 0;
           padded[small*(i+kern_cent_Y)+kern_cent_X+data_size_X] = 0;
           memcpy(padded+(small*(i+kern_cent_Y)+kern_cent_X), in+(i*data_size_X), 4*data_size_X);
       }
    }

    // main convolution loop
    
    #pragma omp parallel
    {
        #pragma omp for private(x,y,i,j,big,out_index,v_a,v_b,v_c,v_d,s_a,s_b,s_c,s_d,kernel_vector)    
        for(y = 0; y < data_size_Y; ++y){ // the x coordinate of the output location we're focusing on
                for(x = 0; x < data_size_X/16*16; x+=16){ // the y coordinate of theoutput location we're focusing on
                                                for(j = -kern_cent_Y; j <= kern_cent_Y; ++j){ // kernel unflipped x coordinate
                                                                big = x + kern_cent_X + (y + kern_cent_Y+j) * small;
                                                                out_index = out + (x + y*data_size_X);
                                for(i = -kern_cent_X; i <= kern_cent_X; ++i){ // kernel unflipped y coordinate
                                        //Note that the kernel is flipped

                                                                                v_a = _mm_loadu_ps((const float*)(i + padded + big));
                                                                                v_b = _mm_loadu_ps((const float*)(i + padded + big + 4));
                                                                                v_c = _mm_loadu_ps((const float*)(i + padded + big + 8));
                                                                                v_d = _mm_loadu_ps((const float*)(i + padded + big + 12));

                                                                                kernel_vector = _mm_set1_ps(kernel[(kern_cent_X-i)+(kern_cent_Y-j)*kernel_x]);

                                                                                v_a = _mm_mul_ps(v_a,kernel_vector);
                                                                                v_b = _mm_mul_ps(v_b,kernel_vector);
                                                                                v_c = _mm_mul_ps(v_c,kernel_vector);
                                                                                v_d = _mm_mul_ps(v_d,kernel_vector);

                                                                                s_a = _mm_add_ps(s_a,v_a);
                                                                                s_b = _mm_add_ps(s_b,v_b);
                                                                                s_c = _mm_add_ps(s_c,v_c);
                                                                                s_d = _mm_add_ps(s_d,v_d);
                                                                }
                        }
                                                _mm_storeu_ps((float*)(out_index), s_a);
                                                _mm_storeu_ps((float*)(out_index+4), s_b);
                                                _mm_storeu_ps((float*)(out_index+8), s_c);
                                                _mm_storeu_ps((float*)(out_index+12), s_d);

                                                s_a = _mm_setzero_ps();
                                                s_b = _mm_setzero_ps();
                                                s_c = _mm_setzero_ps();
                                                s_d = _mm_setzero_ps();
                }

                // secondary loop

         //   for(; x < data_size_X/8*8; x+=8){ // the y coordinate of theoutput location we're focusing on
          //                                      for(j = -kern_cent_Y; j <= kern_cent_Y; ++j){ // kernel unflipped x coordinate
          //                                      big = x + kern_cent_X + (y + kern_cent_Y+j) * small;
          //                                      out_index = out + (x + y*data_size_X);
          //                     for(i = -kern_cent_X; i <= kern_cent_X; ++i){ // kernel unflipped y coordinate
          //                              //Note that the kernel is flipped
//
//                                                                                v_a = _mm_loadu_ps((const float*)(i + padded + big));
//                                                                                v_b = _mm_loadu_ps((const float*)(i + padded + big + 4));
//
 //                                                                               kernel_vector = _mm_set1_ps(kernel[(kern_cent_X-i)+(kern_cent_Y-j)*kernel_x]);
//
 //                                                                               v_a = _mm_mul_ps(v_a,kernel_vector);
  //                                                                              v_b = _mm_mul_ps(v_b,kernel_vector);
//
//                                                                                s_a = _mm_add_ps(s_a,v_a);
 //                                                                               s_b = _mm_add_ps(s_b,v_b);
  //                                                              }
   //                     }
    //                                            _mm_storeu_ps((float*)(out_index), s_a);
     //                                           _mm_storeu_ps((float*)(out_index+4), s_b);
//
 //                                               s_a = _mm_setzero_ps();
  //                                              s_b = _mm_setzero_ps();
   //             }

                // tertiary loop
        for(; x < data_size_X/4*4; x+=4){ // the y coordinate of theoutput location we're focusing on
                                                for(j = -kern_cent_Y; j <= kern_cent_Y; ++j){ // kernel unflipped x coordinate
                                                big = x + kern_cent_X + (y + kern_cent_Y+j) * small;
                                                out_index = out + (x + y*data_size_X);
                                for(i = -kern_cent_X; i <= kern_cent_X; ++i){ // kernel unflipped y coordinate
                                        //Note that the kernel is flipped

                                                                                v_a = _mm_loadu_ps((const float*)(i + padded + big));

                                                                                kernel_vector = _mm_set1_ps(kernel[(kern_cent_X-i)+(kern_cent_Y-j)*kernel_x]);

                                                                                v_a = _mm_mul_ps(v_a,kernel_vector);

                                                                                s_a = _mm_add_ps(s_a,v_a);
                                                                }
                        }
                                                _mm_storeu_ps((float*)(out_index), s_a);

                                                s_a = _mm_setzero_ps();
                                }

                for(;x < data_size_X; ++x){ // the x coordinate of the output location we're focusing on
                        out_index = out+(x+y*data_size_X);
                        for(j = -kern_cent_Y; j <= kern_cent_Y; ++j){ // kernel unflipped y coordinate
                                                                for(i = -kern_cent_X; i <= kern_cent_X; ++i){ // kernel unflipped x coordinate
                                        // only do the operation if not out of bounds
                                        *out_index +=
                                                kernel[(kern_cent_X-i)+(kern_cent_Y-j)*kernel_x] * padded[(x+kern_cent_X+i) + (y+kern_cent_Y+j)*small];
                                }
                        }
                }
            }
        }
        return 1;
}
