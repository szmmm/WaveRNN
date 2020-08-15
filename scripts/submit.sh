qsub -M qd212@cam.ac.uk -m bea -S /bin/bash -l queue_priority=cuda-low,tests=0,mem_grab=0M,gpuclass=*,osrel=*,hostname=air208 ./run_taco_wrnn.sh

# qsub -M qd212@cam.ac.uk -m bea -S /bin/bash -l queue_priority=cuda-low,tests=0,mem_grab=0M,gpuclass=*,osrel=* ./run-aaf-tf.sh
