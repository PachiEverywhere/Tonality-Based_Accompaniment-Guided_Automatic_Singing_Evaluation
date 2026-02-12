import torch
import torch.nn as nn
import torch.nn.functional as F

class KeyEstimationCNN(nn.Module):
    def __init__(self):
        super(KeyEstimationCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(8)
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm2d(8)
        self.conv5 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, padding=2)
        self.bn5 = nn.BatchNorm2d(8)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self._calculate_flatten_size()
        
        self.fc1 = nn.Linear(self.flatten_size, 48)
        self.bn_fc1 = nn.BatchNorm1d(48)
        self.fc2 = nn.Linear(48, 12)
        self.bn_fc2 = nn.BatchNorm1d(12)

        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def _calculate_flatten_size(self):
        sample_input = torch.zeros(1, 1, 103, 175)
        # sample_input = torch.zeros(1, 1, 84, 175)
        sample_output = self._forward_conv_layers(sample_input)
        self.flatten_size = sample_output.view(1, -1).size(1)

    def _forward_conv_layers(self, x):
        x = self.pool(F.elu(self.bn1(self.conv1(x))))
        x = self.pool(F.elu(self.bn2(self.conv2(x))))
        x = self.pool(F.elu(self.bn3(self.conv3(x))))
        x = self.pool(F.elu(self.bn4(self.conv4(x))))
        x = self.pool(F.elu(self.bn5(self.conv5(x))))
        return x
        
    def forward(self, x):
        x = self._forward_conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.softmax(x, dim=1)
        return x



def calculate_mirex_scores(outputs, targets):
    _, predicted = torch.max(outputs, 1)
    total = targets.size(0)

    # Correct (rc)
    rc = (predicted == targets).sum().item() / total

    # Perfect fifth
    fifth_mapping = {
        0: [7],    # C major & A minor -> G major, E minor
        1: [8],    # C# major & D♭ major, A# minor & B♭ minor -> G# major, E♭ minor (G# minor), F minor
        2: [9],    # D major & B minor -> A major, F# minor
        3: [10],  # E♭ major, D# major, C minor -> B♭ major, G minor
        4: [11],  # E major, C# minor, D♭ minor -> B major, G# minor (A♭ minor)
        5: [0],    # F major & D minor -> C major, A minor
        6: [1],    # F# major, G♭ major, D# minor, E♭ minor -> C# major, B♭ minor (D# minor), A# minor
        7: [2],    # G major & E minor -> D major, B minor
        8: [3],    # A♭ major, G# major, F minor -> E♭ major, C minor
        9: [4],    # A major, F# minor, G♭ minor -> E major, C# minor (G♭ minor)
        10: [5],   # B♭ major, A# major, G minor -> F major, D minor
        11: [6]    # B major, G# minor, A♭ minor -> F# major, D# minor
    }

    # Parallel key
    parallel_mapping = {
        0: [3, 9],    # C major & A minor -> C minor, A major
        1: [4, 10],   # C# major & D♭ major, A# minor & B♭ minor -> C# minor, A# major
        2: [5, 11],   # D major & B minor -> D minor, B major
        3: [0, 6],    # E♭ major, D# major, C minor -> C major, E♭ minor
        4: [7, 1],    # E major, C# minor, D♭ minor -> E minor, C# major
        5: [2, 8],    # F major & D minor -> D major, F minor
        6: [9, 3],    # F# major, G♭ major, D# minor, E♭ minor -> F# minor, E♭ major
        7: [10, 4],   # G major & E minor -> G minor, E major
        8: [5, 11],   # A♭ major, G# major, F minor -> F major, A♭ minor
        9: [0, 6],    # A major, F# minor, G♭ minor -> A minor, F# major
        10: [7, 1],   # B♭ major, A# major, G minor -> G major, B♭ minor
        11: [2, 8]    # B major, G# minor, A♭ minor -> B minor, A♭ major
    }


    # Fifth (rf)
    rf = sum(any(pred == ft for ft in fifth_mapping.get(t.item(), [])) for pred, t in zip(predicted, targets)) / total
    # Parallel (rp)
    rp = sum(any(pred == pt for pt in parallel_mapping.get(t.item(), [])) for pred, t in zip(predicted, targets)) / total
    # Since Relative Minor/Major and Parallel Minor/Major have been merged into the same classes,
    # Relative Minor/Major (rr) and Parallel Minor/Major (rp)
    rr = 0 
    weighted_score = rc + 0.5 * rf + 0.2 * rp

    return weighted_score, rc, rf, rr, rp

