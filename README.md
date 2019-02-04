# Repository Overview

1. The `c3d` folder contains the code we implemented in our research. This
   includes the code for OutlineNet (this is the c3d.shape2 package) and
   CareLoss (this is the c3d.smartloss package). The rest of the folders under
   `c3d` are common infrastructure, and are probably not of much interest.

2. The pointnet and PointCNN directories, used for running the baseline
   experiments, contain forks of the original sources for each of these
   projects - https://github.com/charlesq34/pointnet and
   https://github.com/yangyanli/PointCNN.

3. The fork of PointCNN is minor, and is only made to enable us to easily run it
   within our own framework. The full list of changes can be obtained by
   comparing our code to revision 056cff8c of the original repository.

4. The fork of PointNet refactors out some code, to enable us to create the dual
   architecture that handles angles and points separately. Aside from that, we
   made a few small changes such as removing some of the batch normalization
   where it prevented convergence with our non-normalized data. The full list
   of changes can be obtained by comparing our code to revision d64d2398 of the
   original repository.

