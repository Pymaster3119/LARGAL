from pathlib import Path

root = Path('.').resolve()
images_train = root / 'dataset' / 'images' / 'train'
labels_train = root / 'dataset' / 'labels' / 'train'

print('Root:', root)
print('Images train dir:', images_train)
print('Labels train dir:', labels_train)

images = []
if images_train.exists():
    images = [p for p in images_train.rglob('*') if p.is_file()]
labels = []
if labels_train.exists():
    labels = [p for p in labels_train.rglob('*.txt') if p.is_file()]

print('\nCounts:')
print('  images (train):', len(images))
print('  label files (train):', len(labels))

img_stems = {p.stem: p for p in images}
lbl_stems = {p.stem: p for p in labels}

matches = sorted(set(img_stems.keys()) & set(lbl_stems.keys()))
missing_labels = sorted(set(img_stems.keys()) - set(lbl_stems.keys()))
extra_labels = sorted(set(lbl_stems.keys()) - set(img_stems.keys()))

print('\nMatches (image + label) count:', len(matches))
print('Images missing labels count:', len(missing_labels))
print('Label files without image count:', len(extra_labels))

print('\nSample missing label stems (up to 10):')
for s in missing_labels[:10]:
    print(' -', s)

print('\nSample extra label stems (up to 10):')
for s in extra_labels[:10]:
    print(' -', s)

# Inspect label file contents and check formatting
total_instances = 0
bad_lines = []
empty_label_files = []

for lp in labels:
    txt = lp.read_text(encoding='utf-8').strip()
    if not txt:
        empty_label_files.append(str(lp))
        continue
    for i,ln in enumerate(txt.splitlines(), start=1):
        parts = ln.split()
        if len(parts) < 5:
            bad_lines.append((str(lp), i, ln))
        else:
            try:
                cls = int(parts[0])
            except Exception:
                bad_lines.append((str(lp), i, ln))
            total_instances += 1

print('\nTotal instances (label lines):', total_instances)
print('Empty label files count:', len(empty_label_files))
print('Bad-formatted label lines count:', len(bad_lines))

if empty_label_files:
    print('\nExamples of empty label files (up to 5):')
    for p in empty_label_files[:5]:
        print(' -', p)

if bad_lines:
    print('\nExamples of bad-formatted lines (up to 5):')
    for p,i,ln in bad_lines[:5]:
        print(f' - {p} (line {i}): "{ln}"')

# Show a few sample label files content
print('\nSample label files content (up to 5):')
for lp in labels[:5]:
    print('\n---', lp)
    print(lp.read_text(encoding='utf-8'))

# Read dataset.yaml if present
yaml_path = root / 'dataset.yaml'
if yaml_path.exists():
    print('\nFound dataset.yaml:', yaml_path)
    try:
        import yaml
        d = yaml.safe_load(yaml_path.read_text(encoding='utf-8'))
        print('dataset.yaml content:')
        print(d)
    except Exception as e:
        print('Could not parse dataset.yaml:', e)
else:
    print('\nNo dataset.yaml found at', yaml_path)

print('\nDiagnostic complete.')
