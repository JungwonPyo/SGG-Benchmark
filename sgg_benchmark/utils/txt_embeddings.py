import torch
import array
import os
from tqdm import tqdm
import zipfile
import six
import torch
from six.moves.urllib.request import urlretrieve

def obj_edge_vectors(names, wv_dir='', wv_type='glove', wv_dim=300, use_cache=False):
    if 'glove' in wv_type:
        wv_dict, wv_arr, wv_size = load_word_vectors(wv_dir, wv_type, wv_dim)

        vectors = torch.Tensor(len(names), wv_dim)
        vectors.normal_(0,1)

        for i, token in enumerate(names):
            wv_index = wv_dict.get(token, None)
            if wv_index is not None:
                vectors[i] = wv_arr[wv_index]
            else:
                # Try the longest word
                lw_token = sorted(token.split(' '), key=lambda x: len(x), reverse=True)[0]
                # print("{} -> {} ".format(token, lw_token))
                wv_index = wv_dict.get(lw_token, None)
                if wv_index is not None:
                    vectors[i] = wv_arr[wv_index]
        return vectors
    elif wv_type == 'clip':
        # check cache
        if use_cache:
            cache_file = os.path.join(wv_dir, wv_type + '_obj.pt')
            if os.path.exists(cache_file):
                txt_feats = torch.load(cache_file)
                if len(names) == txt_feats.size(0):
                    return txt_feats
        import clip

        model = clip.load("ViT-B/32")[0]
        device = next(model.parameters()).device
        text_token = clip.tokenize(names).to(device)
        txt_feats = [model.encode_text(token).detach() for token in text_token.split(80)]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
        txt_feats = txt_feats.float().cpu()  # always return float32 on cpu

        del model
        torch.cuda.empty_cache()

        # saving to cache
        if use_cache:
            if not os.path.exists(wv_dir):
                os.makedirs(wv_dir)
            torch.save(txt_feats, cache_file)

    elif wv_type == 'siglip':

        if use_cache:
            cache_file = os.path.join(wv_dir, wv_type + '_obj.pt')
            if os.path.exists(cache_file):
                txt_feats = torch.load(cache_file)
                if len(names) == txt_feats.size(0):
                    return txt_feats
                
        from transformers import AutoTokenizer, AutoModel
        model = AutoModel.from_pretrained("google/siglip2-base-patch16-224")
        tokenizer = AutoTokenizer.from_pretrained("google/siglip2-base-patch16-224")

        inputs = tokenizer(names, padding="max_length", return_tensors="pt")
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)

        txt_feats = text_features.detach()

        del model
        torch.cuda.empty_cache()

        # saving to cache
        if use_cache:
            if not os.path.exists(wv_dir):
                os.makedirs(wv_dir)
            torch.save(txt_feats, cache_file)
    else:
        raise ValueError(f"Unknown word vector type: {wv_type}")

    return txt_feats

def rel_vectors(names, wv_dir='', wv_type='clip', wv_dim=300, use_cache=False):
    if 'glove' in wv_type:
        wv_dict, wv_arr, wv_size = load_word_vectors(wv_dir, wv_type, wv_dim)

        vectors = torch.Tensor(len(names), wv_dim)  # 51, 200
        vectors.normal_(0, 1)
        for i, token in enumerate(names):
            if i == 0:
                continue
            wv_index = wv_dict.get(token, None)
            if wv_index is not None:
                vectors[i] = wv_arr[wv_index]
            else:
                split_token = token.split(' ')
                ss = 0
                s_vec = torch.zeros(wv_dim)
                for s_token in split_token:
                    wv_index = wv_dict.get(s_token)
                    if wv_index is not None:
                        ss += 1
                        s_vec += wv_arr[wv_index]
                    else:
                        print("fail on {}".format(token))
                if ss > 0:
                    s_vec /= ss
                    vectors[i] = s_vec
                # else: keeps the normal initialization
        return vectors
    elif wv_type == "clip":
        # check cache
        if use_cache:
            cache_file = os.path.join(wv_dir, wv_type + '_rel.pt')
            if os.path.exists(cache_file):
                txt_feats = torch.load(cache_file, map_location='cpu', weights_only=False).float()
                if len(names) == txt_feats.size(0):
                    return txt_feats
        import clip

        model = clip.load("ViT-B/32")[0]
        device = next(model.parameters()).device
        text_token = clip.tokenize(names).to(device)
        txt_feats = [model.encode_text(token).detach() for token in text_token.split(80)]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
        txt_feats = txt_feats.float().cpu()  # always return float32 on cpu

        del model
        torch.cuda.empty_cache()

        if use_cache:
            if not os.path.exists(wv_dir):
                os.makedirs(wv_dir)
            torch.save(txt_feats, cache_file)

        return txt_feats

### ORIGINAL CODE ###
### DEPRECATED ###


def mpnet_obj_vectors(names, cache_dir='.', d_model=256):
    """
    Encode object class names with all-mpnet-base-v2 (768-dim) and project
    to d_model dims via truncated SVD.  Caches raw 768-dim embeddings to
    <cache_dir>/mpnet_obj.pt so the model download only happens once.

    Returns a [len(names), d_model] float32 tensor with rows L2-normalised
    and scaled to the same expected norm as nn.init.normal_(std=0.1):
        scale = 0.1 * sqrt(d_model)

    The __background__ token (index 0) is zeroed out since it is never a
    real subject or object.
    """
    cache_file = os.path.join(cache_dir, 'mpnet_obj.pt')

    # --- load or compute raw 768-dim embeddings ---
    raw_768 = None
    if os.path.exists(cache_file):
        raw_768 = torch.load(cache_file, map_location='cpu', weights_only=False).float()
        if raw_768.shape[0] != len(names):
            raw_768 = None  # stale cache — recompute

    if raw_768 is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-mpnet-base-v2')
        raw_768 = torch.tensor(
            model.encode(names, batch_size=64, show_progress_bar=False),
            dtype=torch.float32,
        )
        del model
        torch.save(raw_768, cache_file)

    # --- project 768 → d_model via truncated SVD ---
    # Optimal linear reduction: projected = U[:, :k] * S[:k]  (= raw @ Vh[:k].T)
    k = min(d_model, raw_768.shape[1], raw_768.shape[0])
    if k < raw_768.shape[1]:
        U, S, _Vh = torch.linalg.svd(raw_768, full_matrices=False)
        projected = U[:, :k] * S[:k]          # [N, k]
    else:
        projected = raw_768[:, :k].clone()

    # Pad to d_model if k < d_model (e.g. very few classes)
    if k < d_model:
        pad = torch.zeros(projected.shape[0], d_model - k)
        projected = torch.cat([projected, pad], dim=1)

    # Scale so expected L2 norm matches nn.init.normal_(std=0.1)
    scale = 0.1 * (d_model ** 0.5)
    norms = projected.norm(dim=1, keepdim=True).clamp(min=1e-6)
    projected = projected / norms * scale

    # Zero out the __background__ token
    if names[0].lower() in ('__background__', 'background', 'bg'):
        projected[0].zero_()

    return projected

def obj_edge_vectors_glove(names, wv_dir, wv_type='glove.6B', wv_dim=300):
    wv_dict, wv_arr, wv_size = load_word_vectors(wv_dir, wv_type, wv_dim)

    vectors = torch.Tensor(len(names), wv_dim)
    vectors.normal_(0,1)

    for i, token in enumerate(names):
        wv_index = wv_dict.get(token, None)
        if wv_index is not None:
            vectors[i] = wv_arr[wv_index]
        else:
            # Try the longest word
            lw_token = sorted(token.split(' '), key=lambda x: len(x), reverse=True)[0]
            # print("{} -> {} ".format(token, lw_token))
            wv_index = wv_dict.get(lw_token, None)
            if wv_index is not None:
                vectors[i] = wv_arr[wv_index]

    return vectors

def rel_vectors_glove(names, wv_dir, wv_type='glove.6B', wv_dim=300):
    wv_dict, wv_arr, wv_size = load_word_vectors(wv_dir, wv_type, wv_dim)

    vectors = torch.Tensor(len(names), wv_dim)  # 51, 200
    vectors.normal_(0, 1)
    for i, token in enumerate(names):
        if i == 0:
            continue
        wv_index = wv_dict.get(token, None)
        if wv_index is not None:
            vectors[i] = wv_arr[wv_index]
        else:
            split_token = token.split(' ')
            ss = 0
            s_vec = torch.zeros(wv_dim)
            for s_token in split_token:
                wv_index = wv_dict.get(s_token)
                if wv_index is not None:
                    ss += 1
                    s_vec += wv_arr[wv_index]
                else:
                    print("fail on {}".format(token))
            if ss > 0:
                s_vec /= ss
                vectors[i] = s_vec

    return vectors

def load_word_vectors(root, wv_type, dim):
    """Load word vectors from a path, trying .pt, .txt, and .zip extensions."""
    URL = {
        'glove.42B': 'http://nlp.stanford.edu/data/glove.42B.300d.zip',
        'glove.840B': 'http://nlp.stanford.edu/data/glove.840B.300d.zip',
        'glove.twitter.27B': 'http://nlp.stanford.edu/data/glove.twitter.27B.zip',
        'glove.6B': 'http://nlp.stanford.edu/data/glove.6B.zip',
        }
    if isinstance(dim, int):
        dim = str(dim) + 'd'
    fname = os.path.join(root, wv_type + '.' + dim)

    if os.path.isfile(fname + '.pt'):
        fname_pt = fname + '.pt'
        print('loading word vectors from', fname_pt)
        try:
            return torch.load(fname_pt, map_location=torch.device("cpu"))
        except Exception as e:
            print("Error loading the model from {}{}".format(fname_pt, str(e)))
            sys.exit(-1)
    else:
        print("INFO File not found: ", fname + '.pt')
    if not os.path.isfile(fname + '.txt'):
        print("INFO File not found: ", fname + '.txt')
    if os.path.isfile(fname + '.txt'):
        fname_txt = fname + '.txt'
        cm = open(fname_txt, 'rb')
        cm = [line for line in cm]
    elif os.path.basename(wv_type) in URL:
        url = URL[wv_type]
        print('downloading word vectors from {}'.format(url))
        filename = os.path.basename(fname)
        if not os.path.exists(root):
            os.makedirs(root)
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
            fname, _ = urlretrieve(url, fname, reporthook=reporthook(t))
            with zipfile.ZipFile(fname, "r") as zf:
                print('extracting word vectors into {}'.format(root))
                zf.extractall(root)
        if not os.path.isfile(fname + '.txt'):
            raise RuntimeError('no word vectors of requested dimension found')
        return load_word_vectors(root, wv_type, dim)
    else:
        raise RuntimeError('unable to load word vectors')

    wv_tokens, wv_arr, wv_size = [], array.array('d'), None
    if cm is not None:
        for line in tqdm(range(len(cm)), desc="loading word vectors from {}".format(fname_txt)):
            entries = cm[line].strip().split(b' ')
            word, entries = entries[0], entries[1:]
            if wv_size is None:
                wv_size = len(entries)
            try:
                if isinstance(word, six.binary_type):
                    word = word.decode('utf-8')
            except:
                print('non-UTF8 token', repr(word), 'ignored')
                continue
            wv_arr.extend(float(x) for x in entries)
            wv_tokens.append(word)

    wv_dict = {word: i for i, word in enumerate(wv_tokens)}
    wv_arr = torch.Tensor(wv_arr).view(-1, wv_size)
    ret = (wv_dict, wv_arr, wv_size)
    torch.save(ret, fname + '.pt')
    return ret

def reporthook(t):
    """https://github.com/tqdm/tqdm"""
    last_b = [0]
    def inner(b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b
    return inner
