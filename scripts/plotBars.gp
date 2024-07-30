set key autotitle columnhead
set boxwidth 0.5
set style fill solid
set xtics rotate
# set logscale y

set yrange [0:]

plot '<cat' using 1:3:xtic(2) with boxes