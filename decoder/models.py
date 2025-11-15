# Model definitions and mapping for the decoder

# Import decoder model classes
from decoder.decoder_models import FCDecoder, TransformerDecoder

# Model mapping dictionary
decoder_dict = {
    'FCDecoder': FCDecoder,
    'TransformerDecoder': TransformerDecoder,
} 