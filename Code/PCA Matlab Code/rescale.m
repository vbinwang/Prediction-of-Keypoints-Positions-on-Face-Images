function y = rescale(x, new_min, new_max)

old_min = min(x);
old_max = max(x);

y = (new_max-new_min)/(old_max-old_min)*(x-old_max)+new_max;

end

