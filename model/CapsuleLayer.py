import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import parameters

epsilon = 1e-9

class ConvUnit(nn.Module):
    def __init__(self,num_filters=50):
        super(ConvUnit, self).__init__()
        self.conv = nn.Conv2d(1, num_filters, (3, 1))
        weights = torch.FloatTensor(num_filters * [0.1, 0.1, -0.1]).view(num_filters, 1, 3, 1)
        self.conv.weight.data = nn.Parameter(weights)
        self.relu = nn.ReLU()


    def forward(self,x):
        x = self.relu(self.conv(x))
        return x

class CapsuleLayer(nn.Module):
    def __init__(self, vec_len_secondCaps, num_caps_i, num_units, unit_size, use_routing, num_iterations=3, num_filters=50):
        super(CapsuleLayer, self).__init__()

        self.vec_len_secondCaps = vec_len_secondCaps
        self.num_caps_i = num_caps_i
        self.num_units = num_units
        self.use_routing = use_routing
        self.unit_size = unit_size
        self.num_filters = num_filters
        self.num_iterations = num_iterations

        if self.use_routing:
            # In the paper, the deeper capsule layer(s) with capsule inputs (DigitCaps) use a special routing algorithm
            # that uses this weight matrix.
            # W: [1, num_caps_i, num_caps_j, len_u_i, len_v_j]
            # W: [1, k, 1, num_filters, len_v_j (ouput = 10)]
            self.W = nn.Parameter(torch.randn(1, num_caps_i, num_units, unit_size, vec_len_secondCaps))
            #print("W: ",self.W.size())
            # self.W = nn.Parameter(torch.randn(1, 20, 1, 50, 10))
        else:
            # The first convolutional capsule layer (PrimaryCapsules in the paper) does not perform routing.
            # Instead, it is composed of several convolutional units, each of which sees the full input.
            # It is implemented as a normal convolutional layer with a special nonlinearity (squash()).
            def create_conv_unit(unit_idx):
                unit = ConvUnit(num_filters=self.num_filters)
                self.add_module("unit_" + str(unit_idx), unit)
                return unit
            self.units = [create_conv_unit(i) for i in range(self.num_units)]


    @staticmethod
    def squash(s):
        # This is equation 1 from the paper.
        mag_sq = torch.sum(s**2, dim=2, keepdim=True)
        mag = torch.sqrt(mag_sq + epsilon)
        s = (mag_sq / (1.0 + mag_sq)) * (s / mag)
        return s

    def forward(self, x):
        if self.use_routing:
            return self.routing(x)
        else:
            return self.no_routing(x)

    def no_routing(self, x):
        #Using only one Conv2d
        u = self.units[0](x)
        u = CapsuleLayer.squash(u)
        #print("u: ",u.size())
        # Return squashed outputs.
        return u

    def routing(self, x):
        # x: [batch_size, num_filters, 1, k]
        batch_size = x.size(0)
        # print("x: ",x.size())

        # Reshape the input into [batch_size, k, num_filters, 1]
        x = x.view(batch_size,-1,self.num_filters,1)
        #print("x: ", x.size())

        # Reshape the input into [batch_size, k, 1 (= number of conv2d), num_filters, 1]
        x = torch.stack([x] * self.num_units, dim=2)
        #print("x_stack: ",x.size())

        # x = torch.reshape(x,(batch_size,self.num_caps_i,self.num_units,self.unit_size,-1))
        # print(x.size())

        # W: [batch_size, k, 1, num_filters, len_v_j (ouput = 10)]
        W = torch.cat([self.W] * batch_size, dim=0)
        # Transform inputs by weight matrix.
        # W: [batch_size, k, 1, len_v_j (ouput = 10), num_filters]
        W = W.transpose(3,4)
        #print("W: ",W.size())

        #u_hat: (batch_size, k, 1 (=num_units), len_v_j (ouput = 10), 1)
        u_hat = torch.matmul(W, x)
        #print("u_hat: ", u_hat.size())

        # In forward, u_hat_stopped = u_hat; in backward, no gradient passed back from u_hat_stopped to u_hat
        u_hat_stopped = u_hat.detach()
        #print("u_hat_stopped: ", u_hat_stopped.size())

        # Initialize routing logits to zero.
        #b_ij: [batch_size, num_caps_l, num_caps_l_plus_1, 1, 1],
        if torch.cuda.is_available():
            device = torch.device("cuda:"+parameters.DEVICE if torch.cuda.is_available() else "cpu")
            b_ij = Variable(torch.zeros(batch_size, self.num_caps_i, self.num_units, 1, 1)).to(device)
        else:
            b_ij = Variable(torch.zeros(batch_size, self.num_caps_i, self.num_units, 1, 1))

        # Iterative routing.
        for iteration in range(self.num_iterations):
            # Convert routing logits to softmax.
            # c_ij: [batch_size, k, 1 (=num_units), 1, 1]
            c_ij = F.softmax(b_ij,dim=1)
            #print("c_ij_{}: {}".format(iteration,c_ij.size()))

            if iteration == self.num_iterations-1:
                # s_j: [batch_size, k, 1 (=num_units), len_v_j (ouput = 10), 1]
                s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
                #print("s_j: ", s_j.size())

                # v_j: [batch_size, 1, 1 (=num_units), len_v_j (ouput = 10), 1]
                v_j = CapsuleLayer.squash(s_j)
                #print("v_j: ", v_j.size())

            elif iteration < self.num_iterations -1:
                # s_j: [batch_size, k, 1 (=num_units), len_v_j (ouput = 10), 1]
                s_j = (c_ij * u_hat_stopped).sum(dim=1, keepdim=True)
                #print("s_j: ", s_j.size())

                # v_j: [batch_size, k, 1 (=num_units), len_v_j (ouput = 10), 1]
                v_j = CapsuleLayer.squash(s_j)
                #print("v_j: ", v_j.size())

                # reshape & tile v_j from [batch_size ,1, 1, 10, 1] to [batch_size, k, 10, 1, 1]
                v_j1 = torch.cat([v_j] * self.num_caps_i, dim=1)
                #print("v_j1_{}: {}".format(iteration, v_j1.size()))

                # matmul in the last tow dim: [10, 1].T x [10, 1] => [1, 1], reduce mean in the
                # batch_size dim, resulting u_vj1: [1, k, 1 (=num_units), 1, 1]
                u_vj1 = torch.matmul(u_hat_stopped.transpose(3, 4), v_j1).mean(dim=0, keepdim=True)
                #print("u_vj1_{}: {}".format(iteration, u_vj1.size()))

                # Update b_ij(routing)
                b_ij = b_ij + u_vj1

        #v_j: [batch_size, len_v_j (ouput = 10)]
        v_j = v_j.squeeze()
        #print("v_j: ", v_j.size())
        return v_j
