# Fourier-kolam
Kolam (Rangoli) is a traditional South Indian form of line art, preserved by many generations of women. The artwork starts with a grid of dots called __chukki__ which forms the skeleton. A line passes around each __chukki__ only once. The goal is to generate an artwork where every dot is enclosed by the line. A fun challenge is to generate a kolam without lifting the hand in between. Such kolams are called __sikku kolam__ and has attracted the attention of many mathematicians and computer scientists to solve the challenge of one-stroke kolam generation [1,2]. For a given __chukki__ pattern, less than 1% of all possible kolams are __sikku kolam__. 

The loop can be imagined as guided through 4 gates around each dot as shown below. The gates can be either open or closed. The loop passes through an open gate and gets diverted by a closed gate. In the algorithm described in [3] the gating matrix is iteratively tuned so as to obtain a kolam in one stroke. 

In this code, I try to encode fourier amplitudes of an image as a kolam pattern. This is done by considering the __chukki__ pattern as fourier frequencies and the number of open gates around each dot as its amplitude. The amplitudes are quantised before assigning to a particular dot. 



Approximately representing fourier transform of an image as a Kolam

1. Waring, T. M. (2012). Sequential encoding of Tamil kolam patterns. Forma, 27(1), 83-92.
2. Yanagisawa, K. & Nagata, S. Fundamental study on design system of kolam pattern. Forma 22, 31–46 (2007).
3. Sivakumar, S., Sivakumar, S. An algorithm for one-stroke kolam generation using a gating structure. npj Herit. Sci. 14, 144 (2026). https://doi.org/10.1038/s40494-026-02310-3
