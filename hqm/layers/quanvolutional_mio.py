import torchvision
import torch
import sys

sys.path += ['.', './utils/', '/circuits/']

from hqm.circuits.circuit import QuantumCircuit
from hqm.utils.aiinterface import AIInterface

#la differenza è che non aggiungo una dimensione in piu out [(bs, self.filters, h_out, w_out, ch)] ma solo out[(bs, self.filters, h_out, w_out)], alla fine normalozzo dividendo per ch
class Quanvolution2D_mio(torch.nn.Module):
    '''
        Quanvolution2D layer.

        Currently supports only Torch.
    '''

    def __init__(self, qcircuit: QuantumCircuit, filters: int, kernelsize: int = 3, stride: int = 1, padding: str = 'same', aiframework: str = 'torch') -> None:
        '''
        Quanvolution2D constructor.  

        Parameters:  
        -----------  
        - qcircuit: hqm.circuits.circuit.QuantumCircuit  
            QuantumCircuit object to be embedded into the quantum layer
        - filters: int
            number of quanvolution filters
        - kernelsize: int
            size of quanvolution kernel
        - stride: int
            stride for quanvolution operation
        - padding: str
            padding mode, same or valid
        - aiframework: str    
            string representing the AI framework in use, can be 'torch' or 'keras'. This will create  
            a compatible trainable layer for the framework.

        Returns:    
        --------     
        Nothing, a Quanvolution2D object will be created.  
        '''

        super().__init__()

        if aiframework not in ['torch', 'keras']: raise Exception(f"Quanvolution2D currently supports only 'torch' as framework, found {aiframework}")
        if kernelsize < 1: raise Exception(f"kernelsize must be greater than 1, found {kernelsize}")
        if stride < 1: raise Exception(f"stride must be greater than 1, found {stride}")
        
        self.aiframework = aiframework
        self.n_qubits = qcircuit.n_qubits
        
        if kernelsize**2 > self.n_qubits: raise Exception(f"kernelsize**2 must be lower than n_qubits, found kernelsize**2={kernelsize**2} and {self.n_qubits}")
        if (kernelsize**2) * filters > self.n_qubits: raise Exception(f"filters must be lower than n_qubits, found {filters} and {self.n_qubits}")
        self.filters = filters 
        self.kernelsize = kernelsize
        self.stride = stride
        self.padding = padding
        self.qlayer = AIInterface.network_layer(
            circuit=qcircuit.circuit, 
            weight_shape=qcircuit.weight_shape, 
            n_qubits=qcircuit.n_qubits, 
            aiframework=self.aiframework
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Torch forward function for quanvolution layer

        Parameters:
        -----------
        - x: torch.Tensor
            input image or tensor
        
        Returns:
        --------
        - out: torch.Tensor
            quanvoluted input
        '''

        if len(x.shape) != 4: raise Exception(f"x must be a tensor of 4 elements (batch, channels, width, height), found {len(x.shape)}")

        # Calculates the image shape after the convolution
        bs, ch, h, w = x.shape
        
        h_out = int(((h - self.kernelsize) / self.stride) + 1)
        w_out = int(((w - self.kernelsize) / self.stride) + 1)
        
        if self.padding == 'same':
            w_pad = int(round((w - w_out) / 2))
            h_pad = int(round((h - h_out) / 2))
        else:
            h_pad = 0
            w_pad = 0 

        out = torch.zeros((bs, self.filters, h_out, w_out))

        # Batch Loop
        for b in range(bs):
            # Channel Loop
            for c in range(ch):
                # Spatial Loops                                                
                for j in range(0, h_out, self.stride):
                    for k in range(0, w_out, self.stride):            
                        # Process a kernel_size*kernel_size region of the images
                        # with the quantum circuit stride*stride
                        p = x[b, c, j:j+self.kernelsize, k:k+self.kernelsize].reshape(-1)
                        q_results = self.qlayer(p)

                        for f in range(self.filters):
                            out[b, f, j, k] += q_results[f]
        out /= ch 

        if self.padding == 'same':
            out = torch.nn.functional.pad(out, (w_pad, w_pad, h_pad, h_pad), "constant", 0)
            out = torchvision.transforms.Resize([h, w])(out)

        return out
