from vocoder import *

# initialize vocoder object
vocoder_obj = vocoder()

# run the algorithm
vocoder_obj.run()

# export the result
vocoder_obj.export_output('processed_vocale')