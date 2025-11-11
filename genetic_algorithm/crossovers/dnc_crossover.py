import torch
import torch.nn as nn
from typing import Optional, Union
from genetic_algorithm.crossovers.base_crossover import BaseCrossover


# LSTM-based encoder for parent representations
class ParentEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super(ParentEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    
    def forward(self, x: torch.Tensor):
        """
        Encodes parent chromosome into hidden representation.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            output: LSTM output of shape (batch_size, seq_len, hidden_size)
            hidden: Tuple of (h_n, c_n) hidden states
        """
        output, hidden = self.lstm(x)
        return output, hidden


# LSTM-based decoder for offspring generation
class OffspringDecoder(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, num_layers: int = 1):
        super(OffspringDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x: torch.Tensor, hidden):
        """
        Decodes hidden representation into offspring genes.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            hidden: Tuple of (h_n, c_n) hidden states from encoder
            
        Returns:
            output: Gene selection probabilities of shape (batch_size, seq_len, output_size)
        """
        output, hidden = self.lstm(x, hidden)
        output = self.fc(output)
        return output


# DNC (Deep Neural Crossover) - LSTM-based crossover operator
class DNCrossover(BaseCrossover):
    def __init__(
        self,
        gene_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        model_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        seed: Optional[int] = None
    ):
        """
        Deep Neural Crossover operator using LSTM encoder-decoder architecture.
        
        Args:
            gene_size: Size/dimensionality of each gene
            hidden_size: Size of LSTM hidden state
            num_layers: Number of LSTM layers
            model_path: Path to pretrained model weights (optional)
            device: Device to run the model on ('cpu', 'cuda', or torch.device)
            seed: Random seed for reproducibility
        """
        super().__init__(seed)
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        self.gene_size = gene_size
        self.hidden_size = hidden_size
        
        # encoder processes both parents
        self.encoder = ParentEncoder(gene_size, hidden_size, num_layers).to(self.device)
        
        # decoder generates offspring by selecting from parents
        # output size is 2 (probability of selecting from parent1 or parent2)
        self.decoder = OffspringDecoder(hidden_size, 2, num_layers).to(self.device)
        
        # load pretrained weights if provided
        if model_path is not None:
            self.load_model(model_path)
        
        self.encoder.eval()
        self.decoder.eval()
    
    def load_model(self, model_path: str):
        """Loads pretrained model weights"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
    
    def save_model(self, model_path: str):
        """Saves model weights"""
        torch.save({
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'gene_size': self.gene_size,
            'hidden_size': self.hidden_size
        }, model_path)
    
    def _prepare_input(self, parent: list) -> torch.Tensor:
        """
        Converts parent list to tensor format.
        
        Args:
            parent: Parent chromosome as list
            
        Returns:
            Tensor of shape (1, seq_len, gene_size)
        """
        # if genes are scalars, reshape to (seq_len, 1)
        if isinstance(parent[0], (int, float)):
            parent_tensor = torch.tensor([[g] for g in parent], dtype=torch.float32)
        else:
            parent_tensor = torch.tensor(parent, dtype=torch.float32)
        
        # add batch dimension
        return parent_tensor.unsqueeze(0).to(self.device)
    
    def perform(self, parent1: list, parent2: list, **kwargs) -> list:
        """
        Performs DNC crossover using LSTM encoder-decoder.
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            **kwargs: Additional parameters (temperature for softmax sampling)
            
        Returns:
            Offspring chromosome as list
        """
        self._validate_parents(parent1, parent2)
        
        temperature = kwargs.get('temperature', 1.0)
        
        with torch.no_grad():
            # prepare inputs
            p1_tensor = self._prepare_input(parent1)
            p2_tensor = self._prepare_input(parent2)
            
            # encode both parents
            p1_encoded, p1_hidden = self.encoder(p1_tensor)
            p2_encoded, p2_hidden = self.encoder(p2_tensor)
            
            # combine parent encodings (concatenate along hidden dimension)
            # then average to get combined representation
            combined_encoded = (p1_encoded + p2_encoded) / 2
            
            # average hidden states
            h_combined = (p1_hidden[0] + p2_hidden[0]) / 2
            c_combined = (p1_hidden[1] + p2_hidden[1]) / 2
            combined_hidden = (h_combined, c_combined)
            
            # decode to get gene selection probabilities
            logits = self.decoder(combined_encoded, combined_hidden)
            
            # apply temperature scaling and softmax
            probs = torch.softmax(logits / temperature, dim=-1)
            
            # select genes based on probabilities
            offspring = []
            for i in range(len(parent1)):
                # get probability of selecting from each parent
                p1_prob = probs[0, i, 0].item()
                p2_prob = probs[0, i, 1].item()
                
                # select gene from parent with higher probability
                if p1_prob > p2_prob:
                    offspring.append(parent1[i])
                else:
                    offspring.append(parent2[i])
        
        return offspring
    
    def train_mode(self):
        """Sets model to training mode"""
        self.encoder.train()
        self.decoder.train()
    
    def eval_mode(self):
        """Sets model to evaluation mode"""
        self.encoder.eval()
        self.decoder.eval()
