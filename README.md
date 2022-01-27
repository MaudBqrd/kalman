## Kalman filter and statistical learning of the velocity of a one-dimensional car

We present here a two-weeks work using Machine learning to enhance a Kalman filter in order to the measure the velocity of a car which can slide and slip.

The aim of this work is to combine two measurements thanks to a Kalman filter :
* the acceleration of the car $a\_vehicule$ measured by a an inertial measurement unit
* the speed of the wheel $v\_wheel$.

The wheel can slide. This is considered as a supplementary noise for $v\_wheel$ in the Kalman filter.

Further information can be found in the slides contained in the file "Presentation".

We used Colin Pareiller' simulator to generate the data. It can be found at https://github.com/colin-ai/tire-slip-simulator/ or by contacting Colin Pareiller.