% Script to change endianness

load('pd_icbm_normal_1mm_pn5_rf0.12bits.mat')
MyData = swapbytes(MyData);
save('pd_icbm_normal_1mm_pn5_rf0.12bits.mat','MyData')