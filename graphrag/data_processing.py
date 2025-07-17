# data processing
import json
import csv
import os
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from .utils import clean_text, parse_author, extract_year, map_to_domain

def prepare_dataset(full_data_path, output_path, target_distribution, total_papers=300):
    random.seed(42)
    buckets = defaultdict(list)
    target_counts = {domain: round(proportion * total_papers)
                    for domain, proportion in target_distribution.items()}
    storage_limits = {domain: max(10, int(1.5 * target_counts[domain]))
                    for domain in target_distribution}

    with open(full_data_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                paper = json.loads(line)
                domain = map_to_domain(paper.get("categories", ""))
                if domain and len(buckets[domain]) < storage_limits[domain]:
                    buckets[domain].append(paper)
                    if all(len(buckets[d]) >= target_counts[d] for d in target_distribution):
                        break
            except json.JSONDecodeError:
                continue

    sampled = []
    for domain in target_distribution:
        target = target_counts[domain]
        available = len(buckets[domain])
        if available < target:
            print(f"Warning: Only {available} {domain} papers found (wanted {target})")
            if available > 0:
                sampled.extend(buckets[domain])
        else:
            sampled.extend(random.sample(buckets[domain], target))

    random.shuffle(sampled)
    with open(output_path, "w", encoding="utf-8") as f:
        for paper in sampled:
            f.write(json.dumps(paper) + "\n")
    return output_path

def process_paper(paper, writers, seen_entities):
    try:
        paper_id = paper.get('id', '')
        if not paper_id:
            return False

        title = clean_text(paper.get('title', ''))
        abstract = clean_text(paper.get('abstract', ''))
        categories = paper.get('categories', '').split()
        url = f"https://arxiv.org/abs/{paper_id}"

        authors = []
        if 'authors_parsed' in paper and paper['authors_parsed']:
            authors = [parse_author(author) for author in paper['authors_parsed']]
        elif 'authors' in paper and paper['authors']:
            authors = [clean_text(author.strip()) for author in paper['authors'].split(',')]

        versions = paper.get('versions', [])
        year = extract_year([v['created'] for v in versions]) if versions else None
        update_date = paper.get('update_date', '')
        version_count = len(versions)

        writers['papers'].writerow([
            paper_id, title, abstract, year, update_date, url, version_count
        ])

        for author in authors:
            if author and author != "Unknown":
                if author not in seen_entities['authors']:
                    writers['authors'].writerow([author])
                    seen_entities['authors'].add(author)
                writers['authored_edges'].writerow([paper_id, author])

        for category in categories:
            if category:
                if category not in seen_entities['topics']:
                    writers['topics'].writerow([category])
                    seen_entities['topics'].add(category)
                writers['topic_edges'].writerow([paper_id, category])

        return True
    except Exception as e:
        print(f"Error processing paper {paper.get('id', 'unknown')}: {e}")
        return False

def process_papers(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    files = {
        'papers': open(f'{output_dir}/papers.csv', 'w', newline='', encoding='utf-8'),
        'authors': open(f'{output_dir}/authors.csv', 'w', newline='', encoding='utf-8'),
        'topics': open(f'{output_dir}/topics.csv', 'w', newline='', encoding='utf-8'),
        'authored_edges': open(f'{output_dir}/authored_edges.csv', 'w', newline='', encoding='utf-8'),
        'topic_edges': open(f'{output_dir}/topic_edges.csv', 'w', newline='', encoding='utf-8')
    }

    writers = {
        'papers': csv.writer(files['papers']),
        'authors': csv.writer(files['authors']),
        'topics': csv.writer(files['topics']),
        'authored_edges': csv.writer(files['authored_edges']),
        'topic_edges': csv.writer(files['topic_edges'])
    }

    writers['papers'].writerow(['id', 'title', 'abstract', 'year', 'update_date', 'url', 'version_count'])
    writers['authors'].writerow(['id'])
    writers['topics'].writerow(['id'])
    writers['authored_edges'].writerow(['paper_id', 'author_id'])
    writers['topic_edges'].writerow(['paper_id', 'topic_id'])

    seen_entities = {'authors': set(), 'topics': set()}
    counter = 0
    success_count = 0

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                paper = json.loads(line)
                if process_paper(paper, writers, seen_entities):
                    success_count += 1
                counter += 1
            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"Unexpected error: {e}")

    for f in files.values():
        f.close()

    print(f"Processed {counter} lines, {success_count} papers successfully")
    return {
        'papers': f'{output_dir}/papers.csv',
        'authors': f'{output_dir}/authors.csv',
        'topics': f'{output_dir}/topics.csv',
        'authored_edges': f'{output_dir}/authored_edges.csv',
        'topic_edges': f'{output_dir}/topic_edges.csv'
    }
