# Model definitions and mapping for the decoder

# Assuming FCDecoder and TransformerDecoder are available via the decoder package
from decoder_models import FCDecoder, TransformerDecoder # Absolute import

# Model mapping dictionary
decoder_dict = {
    'FCDecoder': FCDecoder,
    'TransformerDecoder': TransformerDecoder,
} 