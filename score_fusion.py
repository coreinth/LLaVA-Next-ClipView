import json
import csv

bert_file = "bert_chapter_results2.csv"

def load_clip_data(clip_file="clip_chapter_results.json"):
    """Load CLIP visual analysis results"""
    with open(clip_file, 'r') as f:
        clip_data = json.load(f)
    return clip_data['chapters']

def load_bert_data(bert_file=bert_file):
    """Load BERT audio analysis results"""
    bert_scores = []
    with open(bert_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            bert_scores.append({
                'score': float(row['score']),
                'start': float(row['start']),
                'end': float(row['end']),
                'transcript': row['transcript']
            })
    return bert_scores

def mmss_to_seconds(mmss):
    """Convert MM:SS to seconds"""
    minutes, seconds = map(int, mmss.split(':'))
    return minutes * 60 + seconds

def seconds_to_mmss(seconds):
    """Convert seconds to MM:SS format"""
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes:02}:{seconds:02}"

def combine_clip_bert_scores(clip_chapters, bert_scores, time_tolerance=10):
    """Combine CLIP visual scores with BERT audio scores"""
    combined_scores = []
    
    for clip_chapter in clip_chapters:
        clip_time = mmss_to_seconds(clip_chapter['timestamp_mmss'])
        
        best_bert_score = 0
        best_bert_match = None
        
        for bert_entry in bert_scores:
            bert_start = bert_entry['start']
            bert_end = bert_entry['end']
            bert_center = (bert_start + bert_end) / 2
            
            if (bert_start <= clip_time <= bert_end or 
                abs(clip_time - bert_center) <= time_tolerance):
                
                if bert_entry['score'] > best_bert_score:
                    best_bert_score = bert_entry['score']
                    best_bert_match = bert_entry
        
        combined_score = {
            'filename': clip_chapter['filename'],
            'timestamp_mmss': clip_chapter['timestamp_mmss'],
            'timestamp_seconds': clip_time,
            'visual_change': clip_chapter['visual_change'],
            'similarity': clip_chapter['similarity'],
            'bert_score': best_bert_score,
            'bert_transcript': best_bert_match['transcript'] if best_bert_match else '',
            'combined_score': (
                clip_chapter['visual_change'] + 
                clip_chapter['similarity'] + 
                best_bert_score * 2
            )
        }
        
        combined_scores.append(combined_score)
    
    return combined_scores

def filter_chapters(chapters, min_gap_seconds=60, max_chapters=10):
    """Filter chapters by distributing them throughout the video"""
    if not chapters:
        return []
    
    sorted_chapters = sorted(chapters, key=lambda x: x['timestamp_seconds'])
    
    if len(sorted_chapters) <= max_chapters:
        filtered = []
        last_time = -min_gap_seconds
        
        for chapter in sorted_chapters:
            current_time = chapter['timestamp_seconds']
            if current_time - last_time >= min_gap_seconds:
                filtered.append(chapter)
                last_time = current_time
        
        return filtered
    
    video_duration = sorted_chapters[-1]['timestamp_seconds']
    segment_duration = video_duration / max_chapters
    
    distributed_chapters = []
    
    for i in range(max_chapters):
        segment_start = i * segment_duration
        segment_end = (i + 1) * segment_duration
        
        segment_chapters = [
            ch for ch in sorted_chapters 
            if segment_start <= ch['timestamp_seconds'] < segment_end
        ]
        
        if not segment_chapters:
            segment_center = segment_start + segment_duration / 2
            closest_chapter = min(
                sorted_chapters,
                key=lambda x: abs(x['timestamp_seconds'] - segment_center)
            )
            if closest_chapter not in distributed_chapters:
                distributed_chapters.append(closest_chapter)
        else:
            best_chapter = max(segment_chapters, key=lambda x: x['combined_score'])
            if best_chapter not in distributed_chapters:
                distributed_chapters.append(best_chapter)
    
    distributed_chapters.sort(key=lambda x: x['timestamp_seconds'])
        
    filtered = []
    last_time = -min_gap_seconds
    
    for chapter in distributed_chapters:
        current_time = chapter['timestamp_seconds']
        if current_time - last_time >= min_gap_seconds:
            filtered.append(chapter)
            last_time = current_time
    
    return filtered[:max_chapters]

def generate_chapter_title(bert_transcript, max_words=4):
    """Generate a chapter title from BERT transcript"""
    if not bert_transcript:
        return "Chapter"
    
    words = bert_transcript.split()
    important_words = []
    skip_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'this', 'that', 'these', 'those', 'a', 'an'}
    
    for word in words[:15]:
        clean_word = word.strip('.,!?";').lower()
        if clean_word not in skip_words and len(clean_word) > 2:
            important_words.append(word.strip('.,!?";'))
            if len(important_words) >= max_words:
                break
    
    return " ".join(important_words) if important_words else "Chapter"

def create_final_chapters():
    """Main function to create final chapter list"""
    
    print("Loading CLIP and BERT data...")
    clip_chapters = load_clip_data()
    bert_scores = load_bert_data()
    
    print("Combining CLIP and BERT scores...")
    combined_chapters = combine_clip_bert_scores(clip_chapters, bert_scores)
    combined_chapters.sort(key=lambda x: x['combined_score'], reverse=True)
    
    print("Filtering chapters...")
    final_chapters = filter_chapters(combined_chapters)
    final_chapters.sort(key=lambda x: x['timestamp_seconds'])
    
    youtube_chapters = []
    for i, chapter in enumerate(final_chapters):
        title = generate_chapter_title(chapter['bert_transcript'])
        youtube_chapters.append({
            "name": title,
            "start": int(chapter['timestamp_seconds'])
        })
    
    with open("compute_pi_chapters.json", "w") as f:
        json.dump(youtube_chapters, f, indent=2)
    
    print(f"Created {len(youtube_chapters)} final chapters")
    print("Results saved file: compute_pi_chapters.json")
    
    return youtube_chapters

if __name__ == "__main__":
    final_chapters = create_final_chapters()
    
    print("\n=== YOUTUBE CHAPTERS ===")
    for chapter in final_chapters:
        print(f'{{ "name": "{chapter["name"]}", "start": {seconds_to_mmss(chapter["start"])} }},')