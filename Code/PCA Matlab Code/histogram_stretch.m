function[img_s] = histogram_stretch(img, lo_bound, hi_bound)

old_min = prctile(img', lo_bound); % in Octave, must use img' instead of img
old_max = prctile(img', hi_bound);

new_min = 0;
new_max = 255;

img_s = (img-old_min) * ((new_max-new_min)/(old_max-old_min)) + new_min;

img_s(img_s < new_min) = new_min;
img_s(img_s > new_max) = new_max;

end