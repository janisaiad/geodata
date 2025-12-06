lambda on larger scale
same for rho 0.01 0.02 0.04 0.05 0.07 0.09 0.12 0.15 0.18 0.20 0.25 0.30


do again with different splatting; keep 64x64 and doing it 


see the ratio of transported over teleported


we just gonna do 3 ablation studies, varying rho sigma




the idea is to show that unbalanced gives artefact and is very not robust for small resolutions, and some tearing, that we resolve with kde estimation

why to use unbalanced ? for image morphing, pixel art represents the max outlier possible, so if this xorks for pixel art then it would work fine for other images

we choose for several sigma and try to find a good sigma a priori


the goal is to show that this  'To ensure the target grid
is fully covered, the kernel standard deviation must satisfy 𝜎(𝑡) ≥12𝛿𝑎𝑣𝑔(𝑡), where 𝛿𝑎𝑣𝑔(𝑡)' is true and works fine


outliers corresponds to white  in the strawberry and the background for salameche


the yellow background is used to form the white outlier

lower green is used and blue+face of salameche merged gives the green strawberry also

other green and other blue become grey

salameche body + gives some yello to make green and become red by getting intensity from yellow

