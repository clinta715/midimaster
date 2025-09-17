from music_theory import MusicTheory

prog = ['I', 'V', 'IV']
total = 0
count = 0

for i in range(len(prog)-1):
    dist = MusicTheory.calculate_harmonic_distance(prog[i], prog[i+1], 'C major')
    total += dist
    count += 1
    print(f'{prog[i]} to {prog[i+1]}: {dist:.3f}')

print(f'Average: {total/count:.3f}')