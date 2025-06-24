import os
import tempfile
import torch
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
from sentence_transformers import SentenceTransformer, util
import PyPDF2
from docx import Document
import nltk
from nltk.tokenize import sent_tokenize
import logging
import traceback
import requests
import json
from datetime import datetime
import telegram
import re
import hashlib
import time
from config import yandexAPI, folder_id, telegramAPI

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

def extract_text_from_pdf(pdf_file):
    try:
        logger.info(f"Начинаем извлечение текста из PDF файла: {pdf_file}")
        text = ""
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        total_pages = len(pdf_reader.pages)
        logger.info(f"Всего страниц в PDF: {total_pages}")
        
        for i, page in enumerate(pdf_reader.pages, 1):
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                logger.info(f"Обработана страница {i} из {total_pages}")
            except Exception as e:
                logger.error(f"Ошибка при обработке страницы {i}: {str(e)}")
                continue
        
        if not text.strip():
            logger.error("Не удалось извлечь текст из PDF файла")
            return None
            
        logger.info(f"Успешно извлечен текст из PDF. Длина текста: {len(text)} символов")
        return text
    except Exception as e:
        logger.error(f"Ошибка при чтении PDF: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def text_from_docx(docx_file):
    try:
        logger.info(f"Начинаем извлечение текста из DOCX файла: {docx_file}")
        doc = Document(docx_file)
        text = ""
        
        for i, paragraph in enumerate(doc.paragraphs, 1):
            if paragraph.text.strip():
                text += paragraph.text + "\n"
            if i % 100 == 0:
                logger.info(f"Обработано {i} параграфов")
        
        if not text.strip():
            logger.error("Не удалось извлечь текст из DOCX файла")
            return None
            
        logger.info(f"Успешно извлечен текст из DOCX. Длина текста: {len(text)} символов")
        return text
    except Exception as e:
        logger.error(f"Ошибка при чтении DOCX: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def process_document(file_path):
    try:
        logger.info(f"Начинаем обработку документа: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"Файл не найден: {file_path}")
            return None
            
        if file_path.endswith('.pdf'):
            return extract_text_from_pdf(file_path)
        elif file_path.endswith(('.doc', '.docx')):
            return text_from_docx(file_path)
        else:
            logger.error(f"Неподдерживаемый формат файла: {file_path}")
            return None
    except Exception as e:
        logger.error(f"Ошибка при обработке документа: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def clean_pdf(text):
    text = ' '.join(text.split())
    text = re.sub(r'\.+', '.', text)
    text = re.sub(r'\s+[а-яА-Яa-zA-Z0-9]\s+', ' ', text)
    text = re.sub(r'[^а-яА-Яa-zA-Z0-9\s.,!?;:()\-]', '', text)
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def valid_sentence(sentence):

    if len(sentence) < 10:
        return False
    if not any(c.isalpha() for c in sentence):
        return False
    if sentence.count(',') > 5 or sentence.count(';') > 3:
        return False
    if sentence.isupper() and len(sentence) < 50:
        return False
    if re.search(r'[A-Za-z]+\s*=\s*[A-Za-z0-9]+', sentence):
        return True 
    if re.search(r'\b[A-Z]{2,}\b', sentence):
        return True 
    return True

def calculate_importance(sentence, position, total_sentences):

    importance = 0.0
    # Позиция предложения
    if position < total_sentences * 0.2 or position > total_sentences * 0.8:
        importance += 0.3  
    # Длина предложения
    length = len(sentence.split())
    if 10 <= length <= 40:
        importance += 0.2
    # Ключевые слова
    key_terms = [
        'however', 'therefore', 'conclude', 'result', 'important', 'significant',
        'study', 'research', 'analysis', 'method', 'approach', 'findings',
        'demonstrate', 'show', 'prove', 'establish', 'identify', 'determine',
        'investigate', 'examine', 'analyze', 'evaluate', 'assess', 'consider',
        'suggest', 'indicate', 'reveal', 'observe', 'note', 'find',
        'conclusion', 'summary', 'overview', 'introduction', 'background',
        'objective', 'aim', 'goal', 'purpose', 'main', 'primary', 'key'
    ]

    for term in key_terms:
        if term.lower() in sentence.lower():
            importance += 0.15
    if re.search(r'[A-Za-z]+\s*=\s*[A-Za-z0-9]+', sentence):
        importance += 0.25
    if re.search(r'\d+%|\d+\.\d+|\d+\s*/\s*\d+', sentence):
        importance += 0.2
    if re.search(r'is\s+defined|means|refers\s+to|consists\s+of', sentence.lower()):
        importance += 0.2
    return importance

def summarize_pdf(text, max_length=250, min_length=50):
    try:
        text = clean_pdf(text)
        sentences = []
        current_sentence = []

        for line in text.split('.'):
            line = line.strip()
            if not line:
                continue
            
            if len(line) > 3 and not line.isupper() and not any(c.isdigit() for c in line):
                current_sentence.append(line)
                
                if len(' '.join(current_sentence)) > 15:
                    full_sentence = ' '.join(current_sentence)
                    if valid_sentence(full_sentence):
                        sentences.append(full_sentence)
                    current_sentence = []
        
        if current_sentence:
            full_sentence = ' '.join(current_sentence)
            if valid_sentence(full_sentence):
                sentences.append(full_sentence)
        
        if len(sentences) <= 3:
            return text
    
        embeddings = model.encode(sentences, convert_to_tensor=True)
        similarity_matrix = util.cos_sim(embeddings, embeddings)
        
        sentence_scores = []
        for i in range(len(sentences)):
            similar_scores = similarity_matrix[i][similarity_matrix[i] > 0.4]
            if len(similar_scores) > 0:
                base_score = torch.mean(similar_scores)
            else:
                base_score = torch.mean(similarity_matrix[i])
            
            importance_score = calculate_importance(sentences[i], i, len(sentences))
            final_score = float(base_score) + importance_score
            sentence_scores.append((final_score, i))
        sentence_scores.sort(reverse=True)
        num_sentences = max(min_length // 8, min(len(sentences), max_length // 8))
        selected_indices = sorted([idx for _, idx in sentence_scores[:num_sentences]])
        summary_text = ' '.join([sentences[i] for i in selected_indices])
        summary_text = clean_pdf(summary_text)
        summary = f"Краткая выжимка:\n\n{summary_text}"
        
        return summary
    except Exception as e:
        logger.error(f"Ошибка при создании краткой выжимки PDF: {str(e)}")
        logger.error(traceback.format_exc())
        return text

def summarize_docx(text, max_length=130, min_length=30):
    try:
        sentences = sent_tokenize(text)
        valid_sentences = [s for s in sentences if valid_sentence(s)]
        
        if len(valid_sentences) <= 3:
            return text
            
        embeddings = model.encode(valid_sentences, convert_to_tensor=True)
        similarity_matrix = util.cos_sim(embeddings, embeddings)
        sentence_scores = []
        for i in range(len(valid_sentences)):
            score = torch.mean(similarity_matrix[i])
            importance = calculate_importance(valid_sentences[i], i, len(valid_sentences))
            final_score = float(score) + importance
            sentence_scores.append((final_score, i))
        
        sentence_scores.sort(reverse=True)
        num_sentences = max(min_length // 10, min(len(valid_sentences), max_length // 10))
        selected_indices = sorted([idx for _, idx in sentence_scores[:num_sentences]])
        summary_text = ' '.join([valid_sentences[i] for i in selected_indices])
        summary = f"Краткая выжимка:\n\n{summary_text}"
        
        return summary
    except Exception as e:
        logger.error(f"Ошибка при создании краткой выжимки DOCX: {str(e)}")
        logger.error(traceback.format_exc())
        return text

def search(query, year_from=None, year_to=None, limit=10):
    max_retries = 3  
    timeout = 30     
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Поиск статей на arXiv (попытка {attempt + 1}/{max_retries}): query={query}, year_from={year_from}, year_to={year_to}")
            base_url = "http://export.arxiv.org/api/query"
            
            params = {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": limit * 2,  # Увеличиваем лимит для учета фильтрации
                "sortBy": "submittedDate",
                "sortOrder": "descending"
            }
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            response = requests.get(
                base_url, 
                params=params, 
                headers=headers, 
                timeout=timeout
            )
            response.raise_for_status()
            
            # Парсим XML ответ
            from xml.etree import ElementTree as ET
            root = ET.fromstring(response.content)
            
            ns = {'atom': 'http://www.w3.org/2005/Atom',
                  'arxiv': 'http://arxiv.org/schemas/atom'}
            # Извлекаем статьи
            entries = root.findall('.//atom:entry', ns)
            if not entries:
                logger.warning("Статьи не найдены в ответе API arXiv")
                return None  
            articles = []
            for entry in entries:
                try:
                    # Получаем основную информацию
                    title = entry.find('atom:title', ns).text.strip()
                    summary = entry.find('atom:summary', ns).text.strip()
                    published = entry.find('atom:published', ns).text
                    link = entry.find('atom:link[@title="pdf"]', ns).get('href')
                    year = int(published[:4])
                    if year_from and year < year_from:
                        continue
                    if year_to and year > year_to:
                        continue

                    authors = []
                    for author in entry.findall('atom:author/atom:name', ns):
                        authors.append(author.text)

                    categories = []
                    for category in entry.findall('atom:category', ns):
                        categories.append(category.get('term'))
                    
                    article = {
                        "title": title,
                        "authors": ", ".join(authors) or "Авторы не указаны",
                        "abstract": summary,
                        "url": link,
                        "year": str(year),
                        "categories": ", ".join(categories),
                        "venue": "arXiv",
                        "source": "arXiv",
                        "published_date": published
                    }
                    articles.append(article)
                    
                except Exception as e:
                    logger.error(f"Ошибка при обработке статьи из arXiv: {str(e)}")
                    continue
                
            if not articles:
                logger.warning("Не удалось извлечь информацию из найденных статей arXiv")
                return None
                
            # Сортируем по дате публикации
            articles.sort(key=lambda x: x['published_date'], reverse=True)
            articles = articles[:limit]
                
            logger.info(f"Успешно обработано {len(articles)} статей из arXiv")
            return articles
            
        except requests.exceptions.Timeout:
            logger.warning(f"Таймаут при попытке {attempt + 1}/{max_retries}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5  # Увеличиваем время ожидания с каждой попыткой
                logger.info(f"Ожидание {wait_time} секунд перед следующей попыткой...")
                time.sleep(wait_time)
            else:
                logger.error("Превышено максимальное количество попыток")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка при выполнении запроса к API arXiv: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5
                logger.info(f"Ожидание {wait_time} секунд перед следующей попыткой...")
                time.sleep(wait_time)
            else:
                return None
                
        except ET.ParseError as e:
            logger.error(f"Ошибка при разборе XML ответа arXiv: {str(e)}")
            return None
            
        except Exception as e:
            logger.error(f"Неожиданная ошибка при поиске статей на arXiv: {str(e)}")
            logger.error(traceback.format_exc())
            return None

def format_article(article):
    return (
        f"📚 {article['title']}\n\n"
        f"👥 Авторы: {article['authors']}\n"
        f"📅 Год: {article['year']}\n"
        f"🏢 Категории: {article.get('categories', 'Не указаны')}\n"
        f"📝 Источник: arXiv\n"
        f"\n📝 Аннотация:\n{article['abstract']}\n\n"
        f"🔗 Ссылка: {article['url']}\n"
        f"{'─' * 50}\n"
    )

def create_menu():
    keyboard = [
        [InlineKeyboardButton("🔍 Поиск", callback_data='search'),
         InlineKeyboardButton("📝 Краткая выжимка", callback_data='summary')],
        [InlineKeyboardButton("📚 Поиск научных статей", callback_data='scientific_search'),
         InlineKeyboardButton("🌐 Перевод", callback_data='translate')]
    ]
    return InlineKeyboardMarkup(keyboard)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    reply_markup = create_menu()
    await update.message.reply_text(
        "Привет! Я бот для работы с документами и поиска научных статей. Выберите действие:",
        reply_markup=reply_markup
    )

async def menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    reply_markup = create_menu()
    await update.message.reply_text(
        "Выберите действие:",
        reply_markup=reply_markup
    )

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    if query.data == 'search':
        context.user_data['mode'] = 'search'
        await query.message.reply_text("Отправьте документ (PDF или DOC/DOCX), а затем введите поисковый запрос.")
    elif query.data == 'summary':
        context.user_data['mode'] = 'summary'
        await query.message.reply_text("Отправьте документ (PDF или DOC/DOCX) для создания краткой выжимки.")
    elif query.data == 'scientific_search':
        context.user_data['mode'] = 'scientific_search_arxiv'
        # Сохраняем состояние для пошагового ввода
        context.user_data['search_step'] = 'query'
        await query.message.reply_text(
            "🔍 Введите поисковый запрос.\n\n"
            "Например:\n"
            "- machine learning\n"
            "- quantum computing\n"
            "- artificial intelligence\n"
            "- deep learning"
        )
    elif query.data == 'translate':
        context.user_data['mode'] = 'translate'
        await query.message.reply_text(
            "Отправьте текст на английском языке для перевода на русский.\n"
            "Вы можете отправить:\n"
            "1. Текстовое сообщение\n"
            "2. Документ (PDF или DOC/DOCX)"
        )
    elif query.data.startswith('year_range_'):
        # Обработка выбора диапазона лет
        range_type = query.data.split('_')[2]
        if range_type == 'custom':
            context.user_data['search_step'] = 'year_from'
            await query.message.reply_text(
                "Введите начальный год (например, 2020):"
            )
        else:
            current_year = datetime.now().year
            if range_type == 'last1':
                year_from = current_year - 1
                year_to = current_year
            elif range_type == 'last3':
                year_from = current_year - 3
                year_to = current_year
            elif range_type == 'last5':
                year_from = current_year - 5
                year_to = current_year
            elif range_type == 'last10':
                year_from = current_year - 10
                year_to = current_year
            else:  # all
                year_from = None
                year_to = None
                
            context.user_data['year_from'] = year_from
            context.user_data['year_to'] = year_to
            await start_search(update, context)
    elif query.data == 'start_search':
        await start_search(update, context)

async def start_search(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Запускает поиск с сохраненными параметрами"""
    query = context.user_data.get('search_query')
    year_from = context.user_data.get('year_from')
    year_to = context.user_data.get('year_to')
    
    await update.callback_query.message.reply_text("🔍 Ищу статьи на arXiv...")
    
    try:
        articles = search(
            query=query,
            year_from=year_from,
            year_to=year_to
        )
        
        if not articles:
            await update.callback_query.message.reply_text(
                "К сожалению, не удалось найти статьи на arXiv.\n"
                "Попробуйте:\n"
                "1. Использовать английские ключевые слова\n"
                "2. Сделать запрос более конкретным\n"
                "3. Расширить временной диапазон\n"
                "4. Использовать научные термины"
            )
            return
            
        response = f"📚 Найдено {len(articles)} статей на arXiv:\n\n"
        for article in articles:
            response += format_article(article)
            
        # Разбиваем ответ на части
        max_length = 4096
        for i in range(0, len(response), max_length):
            await update.callback_query.message.reply_text(response[i:i + max_length])
            
    except Exception as e:
        logger.error(f"Ошибка при поиске на arXiv: {str(e)}")
        await update.callback_query.message.reply_text(
            "Произошла ошибка при поиске статей. Пожалуйста, попробуйте позже."
        )

def delete_file(file_path, max_retries=3, delay=1):
    for attempt in range(max_retries):
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                logger.info(f"Временный файл успешно удален: {file_path}")
                return True
        except Exception as e:
            logger.warning(f"Попытка {attempt + 1}/{max_retries} удаления файла не удалась: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(delay)
    logger.error(f"Не удалось удалить временный файл после {max_retries} попыток: {file_path}")
    return False

async def send_large_text(update: Update, text: str, prefix: str = "", max_chunk_size: int = 4000) -> bool:
    try:
        # Если текст небольшой, отправляем частями
        if len(text) <= max_chunk_size * 10:
            chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
            total_chunks = len(chunks)
            
            for i, chunk in enumerate(chunks, 1):
                header = f"{prefix}Часть {i} из {total_chunks}\n\n" if total_chunks > 1 else prefix
                try:
                    await update.message.reply_text(header + chunk)
                except Exception as e:
                    logger.error(f"Ошибка при отправке части {i}: {str(e)}")
                    try:
                        await update.message.reply_text(chunk)
                    except Exception as e:
                        logger.error(f"Не удалось отправить часть {i}: {str(e)}")
                        return False
            return True
        else:
            try:
                with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.txt', delete=False) as temp_file:
                    temp_file.write(text)
                    temp_file_path = temp_file.name

                with open(temp_file_path, 'rb') as file:
                    await update.message.reply_document(
                        document=file,
                        filename='translation.txt',
                        caption=f"{prefix}Перевод отправлен в виде файла из-за большого размера текста."
                    )
                delete_file(temp_file_path)
                return True
                
            except Exception as e:
                logger.error(f"Ошибка при отправке файла: {str(e)}")
                if os.path.exists(temp_file_path):
                    delete_file(temp_file_path)
                return False
                
    except Exception as e:
        logger.error(f"Ошибка при отправке текста: {str(e)}")
        return False

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if 'mode' not in context.user_data:
            logger.warning("Попытка обработки документа без выбранного режима")
            await update.message.reply_text("Пожалуйста, сначала выберите режим работы.")
            return

        file = await context.bot.get_file(update.message.document)
        file_name = update.message.document.file_name
        
        logger.info(f"Начало обработки файла: {file_name}, режим: {context.user_data['mode']}")
        
        if not file_name.endswith(('.pdf', '.doc', '.docx')):
            logger.warning(f"Получен неподдерживаемый формат файла: {file_name}")
            await update.message.reply_text("Пожалуйста, отправьте файл в формате PDF или DOC/DOCX.")
            return

        temp_file = None
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1])
            logger.info(f"Создан временный файл: {temp_file.name}")
            await file.download_to_drive(temp_file.name)
            temp_file.close()
            logger.info("Файл успешно скачан")
            logger.info("Начало извлечения текста из документа")
            text = process_document(temp_file.name)
            delete_file(temp_file.name)
            
        except Exception as e:
            logger.error(f"Ошибка при обработке файла: {str(e)}")
            logger.error(traceback.format_exc())
            if temp_file and os.path.exists(temp_file.name):
                delete_file(temp_file.name)
            raise

        if not text:
            logger.error("Не удалось извлечь текст из документа")
            await update.message.reply_text("Не удалось обработать файл. Пожалуйста, проверьте формат и попробуйте снова.")
            return

        logger.info(f"Текст успешно извлечен из файла. Длина текста: {len(text)} символов")
        context.user_data['document_text'] = text
        
        if context.user_data['mode'] == 'search':
            logger.info("Режим поиска: ожидание поискового запроса")
            await update.message.reply_text("Теперь введите поисковый запрос:")
        elif context.user_data['mode'] == 'translate':
            logger.info("Начало перевода документа")
            
            progress_message = await update.message.reply_text(
                "🔄 Перевод документа...\n"
                "⏳ Подготовка к переводу..."
            )
            
            try:
                translated_text = translate_text(text)
                if translated_text:
                    logger.info(f"Перевод успешно выполнен. Длина переведенного текста: {len(translated_text)} символов")
                    await progress_message.edit_text(
                        "✅ Перевод завершен!\n"
                        "⏳ Отправка результата..."
                    )

                    prefix = f"📚 Перевод файла: {file_name}\n\n"
                    if await send_large_text(update, translated_text, prefix):
                        await progress_message.edit_text(
                            "✅ Перевод успешно отправлен!\n"
                            f"📚 Исходный файл: {file_name}"
                        )
                    else:
                        await progress_message.edit_text(
                            "❌ Ошибка при отправке перевода.\n"
                            "Пожалуйста, попробуйте разбить документ на меньшие части."
                        )
                else:
                    logger.error("Ошибка при переводе: пустой результат")
                    await progress_message.edit_text(
                        "❌ Произошла ошибка при переводе документа.\n"
                        "Пожалуйста, попробуйте снова или разбейте документ на меньшие части."
                    )
            except Exception as e:
                logger.error(f"Ошибка при переводе документа: {str(e)}")
                logger.error(traceback.format_exc())
                await progress_message.edit_text(
                    "❌ Произошла ошибка при переводе.\n"
                    "Пожалуйста, попробуйте позже или разбейте документ на меньшие части."
                )
        else:
            logger.info("Начало создания краткой выжимки")
            await update.message.reply_text("Создаю краткую выжимку...")
            try:
                if file_name.endswith('.pdf'):
                    summary = summarize_pdf(text)
                else:
                    summary = summarize_docx(text)
                
                logger.info(f"Краткая выжимка создана. Длина: {len(summary)} символов")
                
                # Разбиваем сообщение на части
                max_length = 4000
                for i in range(0, len(summary), max_length):
                    chunk = summary[i:i + max_length]
                    try:
                        logger.info(f"Отправка части выжимки {i//max_length + 1}")
                        await update.message.reply_text(chunk)
                    except Exception as e:
                        logger.error(f"Ошибка при отправке части выжимки: {str(e)}")
                        shorter_chunk = chunk[:3500]
                        await update.message.reply_text(shorter_chunk)
                logger.info("Краткая выжимка успешно отправлена")
            except Exception as e:
                logger.error(f"Ошибка при создании краткой выжимки: {str(e)}")
                logger.error(traceback.format_exc())
                await update.message.reply_text("Произошла ошибка при создании краткой выжимки. Пожалуйста, попробуйте снова.")
            
    except Exception as e:
        logger.error(f"Критическая ошибка при обработке документа: {str(e)}")
        logger.error(traceback.format_exc())
        await update.message.reply_text("Произошла ошибка при обработке файла. Пожалуйста, попробуйте снова.")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if 'mode' not in context.user_data:
        await update.message.reply_text("Пожалуйста, сначала выберите режим работы.")
        return

    if context.user_data['mode'] == 'translate':
        text = update.message.text
        await update.message.reply_text("🔄 Перевод текста...")
        
        try:
            translated_text = translate_text(text)
            if translated_text:
                # Разбиваем сообщение на части
                max_length = 4000
                for i in range(0, len(translated_text), max_length):
                    chunk = translated_text[i:i + max_length]
                    await update.message.reply_text(chunk)
            else:
                await update.message.reply_text("Произошла ошибка при переводе текста. Пожалуйста, попробуйте снова.")
        except Exception as e:
            logger.error(f"Ошибка при переводе текста: {str(e)}")
            await update.message.reply_text("Произошла ошибка при переводе. Пожалуйста, попробуйте позже.")
        return

    if context.user_data['mode'] == 'scientific_search_arxiv':
        if 'search_step' not in context.user_data:
            context.user_data['search_step'] = 'query'
            
        if context.user_data['search_step'] == 'query':
            context.user_data['search_query'] = update.message.text
            
            keyboard = [
                [InlineKeyboardButton("За последний год", callback_data='year_range_last1')],
                [InlineKeyboardButton("За последние 3 года", callback_data='year_range_last3')],
                [InlineKeyboardButton("За последние 5 лет", callback_data='year_range_last5')],
                [InlineKeyboardButton("За последние 10 лет", callback_data='year_range_last10')],
                [InlineKeyboardButton("За все время", callback_data='year_range_all')],
                [InlineKeyboardButton("Указать свой диапазон", callback_data='year_range_custom')]
            ]
            await update.message.reply_text(
                "Выберите временной диапазон для поиска:",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            
        elif context.user_data['search_step'] == 'year_from':
            try:
                year_from = int(update.message.text)
                if year_from < 1991 or year_from > datetime.now().year:
                    await update.message.reply_text(
                        f"Пожалуйста, введите год между 1991 и {datetime.now().year}:"
                    )
                    return
                    
                context.user_data['year_from'] = year_from
                context.user_data['search_step'] = 'year_to'
                await update.message.reply_text(
                    f"Введите конечный год (от {year_from} до {datetime.now().year}):"
                )
            except ValueError:
                await update.message.reply_text(
                    "Пожалуйста, введите корректный год (например, 2020):"
                )
                
        elif context.user_data['search_step'] == 'year_to':
            try:
                year_to = int(update.message.text)
                year_from = context.user_data['year_from']
                
                if year_to < year_from or year_to > datetime.now().year:
                    await update.message.reply_text(
                        f"Пожалуйста, введите год между {year_from} и {datetime.now().year}:"
                    )
                    return
                    
                context.user_data['year_to'] = year_to
            
                keyboard = [[InlineKeyboardButton("🔍 Начать поиск", callback_data='start_search')]]
                await update.message.reply_text(
                    f"Параметры поиска:\n"
                    f"📝 Запрос: {context.user_data['search_query']}\n"
                    f"📅 Годы: {year_from} - {year_to}\n\n"
                    f"Нажмите кнопку для начала поиска:",
                    reply_markup=InlineKeyboardMarkup(keyboard)
                )
            except ValueError:
                await update.message.reply_text(
                    "Пожалуйста, введите корректный год (например, 2023):"
                )
    elif context.user_data['mode'] == 'search':
        if 'document_text' not in context.user_data:
            await update.message.reply_text("Пожалуйста, сначала отправьте документ.")
            return
            
        query = update.message.text
        text = context.user_data['document_text']
        
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        embeddings = model.encode(sentences, convert_to_tensor=True)
        query_embedding = model.encode(query, convert_to_tensor=True)
        
        cos_scores = util.cos_sim(query_embedding, embeddings)[0]
        top_results = torch.topk(cos_scores, k=min(5, len(sentences)))
        
        response = "Результаты поиска:\n\n"
        for score, idx in zip(top_results[0], top_results[1]):
            if score > 0.3:
                response += f"Релевантность: {score:.2f}\n"
                response += f"Текст: {sentences[idx]}\n"
                response += "---\n"
        
        await update.message.reply_text(response)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /help"""
    help_text = (
        "🤖 Доступные команды:\n\n"
        "/start - Начать работу с ботом\n"
        "/menu - Показать меню действий\n"
        "/help - Показать это сообщение\n"
        "/search - Поиск по документам\n"
        "/summary - Создать краткую выжимку\n"
        "/scientific - Поиск научных статей\n"
        "/translate - Перевод текста\n\n"
        "📝 Как использовать:\n"
        "1. Выберите нужную команду из меню\n"
        "2. Следуйте инструкциям бота\n"
        "3. Для возврата в меню используйте /menu"
    )
    await update.message.reply_text(help_text)

async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['mode'] = 'search'
    await update.message.reply_text("Отправьте документ (PDF или DOC/DOCX), а затем введите поисковый запрос.")

async def summary_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['mode'] = 'summary'
    await update.message.reply_text("Отправьте документ (PDF или DOC/DOCX) для создания краткой выжимки.")

async def scientific_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['mode'] = 'scientific_search_arxiv'
    await update.message.reply_text(
        "🔍 Введите поисковый запрос в формате:\n"
        "запрос | год_от | год_до\n\n"
        "Например:\n"
        "machine learning | 2020 | 2023\n"
        "quantum computing | 2019 | 2023\n"
        "artificial intelligence | 2018 | 2023\n\n"
        "Если не нужно указывать годы, просто введите запрос."
    )

async def translate_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['mode'] = 'translate'
    await update.message.reply_text(
        "Отправьте текст на английском языке для перевода на русский.\n"
        "Вы можете отправить:\n"
        "1. Текстовое сообщение\n"
        "2. Документ (PDF или DOC/DOCX)"
    )

async def setup_commands(application: Application):
    commands = [
        BotCommand("start", "Начать работу с ботом"),
        BotCommand("menu", "Показать меню действий"),
        BotCommand("help", "Показать список команд"),
        BotCommand("search", "Поиск по документам"),
        BotCommand("summary", "Создать краткую выжимку"),
        BotCommand("scientific", "Поиск научных статей"),
        BotCommand("translate", "Перевод текста")
    ]
    await application.bot.set_my_commands(commands)

def translate_text(text, target_lang='ru'):
    try:
        logger.info("Начало процесса перевода")
        if not yandexAPI:
            logger.error("API ключ Яндекс Переводчика не настроен")
            return None
            
        if not folder_id:
            logger.error("ID папки Яндекс Облака не настроен")
            return None

        chr_reqest = 9500  # Максимальное количество символов в одном запросе
        MAX_REQUESTS_PER_MINUTE = 20   # Максимальное количество запросов в минуту
        delay = 3.0                # Минимальная задержка между запросами
        logger.info("Разбиение текста на предложения")
        sentences = sent_tokenize(text)
        logger.info(f"Текст разбит на {len(sentences)} предложений")
        parts = []
        current_part = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            if sentence_length > chr_reqest:
                if current_part:
                    parts.append(' '.join(current_part))
                    current_part = []
                    current_length = 0

                words = sentence.split()
                current_words = []
                current_words_length = 0

                for word in words:
                    word_length = len(word) + 1  # +1 для пробела
                    if current_words_length + word_length > chr_reqest:
                        if current_words:
                            parts.append(' '.join(current_words))
                        current_words = [word]
                        current_words_length = word_length
                    else:
                        current_words.append(word)
                        current_words_length += word_length
                
                if current_words:
                    parts.append(' '.join(current_words))
                continue
            
            if current_length + sentence_length > chr_reqest:
                if current_part:
                    parts.append(' '.join(current_part))
                current_part = [sentence]
                current_length = sentence_length
            else:
                current_part.append(sentence)
                current_length += sentence_length
        
        if current_part:
            parts.append(' '.join(current_part))
            
        logger.info(f"Текст разбит на {len(parts)} частей для перевода")
        
        translated_parts = []
        total_parts = len(parts)
        
        # API Яндекс Переводчика
        url = "https://translate.api.cloud.yandex.net/translate/v2/translate"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {yandexAPI}"
        }
        # Обрабатываем каждую часть
        for i, part in enumerate(parts, 1):
            logger.info(f"Перевод части {i}/{total_parts} (длина: {len(part)} символов)")
            
            if len(part) > chr_reqest:
                logger.warning(f"Часть {i} превышает лимит символов ({len(part)} > {chr_reqest})")
                # Разбиваем часть на более мелкие части
                sub_parts = [part[j:j + chr_reqest] for j in range(0, len(part), chr_reqest)]
                for j, sub_part in enumerate(sub_parts, 1):
                    logger.info(f"Перевод подчасти {j}/{len(sub_parts)} части {i}")
                    try:
                        body = {
                            "targetLanguageCode": target_lang,
                            "texts": [sub_part],
                            "folderId": folder_id
                        }
                        
                        response = requests.post(url, headers=headers, json=body, timeout=30)
                        if response.status_code == 200:
                            translation = response.json()
                            if 'translations' in translation and translation['translations']:
                                translated_parts.append(translation['translations'][0]['text'])
                                logger.info(f"Подчасть {j} части {i} успешно переведена")
                            else:
                                logger.error(f"Неожиданный формат ответа API для подчасти {j} части {i}")
                        else:
                            logger.error(f"Ошибка API при переводе подчасти {j} части {i}: {response.text}")
                    except Exception as e:
                        logger.error(f"Ошибка при переводе подчасти {j} части {i}: {str(e)}")
                        continue
                    # Задержка между запросами
                    if j < len(sub_parts):
                        time.sleep(delay)
                continue
            
            body = {
                "targetLanguageCode": target_lang,
                "texts": [part],
                "folderId": folder_id
            }
            
            try:
                logger.debug(f"Отправка запроса к API для части {i}")
                response = requests.post(url, headers=headers, json=body, timeout=30)
                
                if response.status_code != 200:
                    logger.error(f"Ошибка API (статус {response.status_code}): {response.text}")
                    if response.status_code == 401:
                        logger.error("Ошибка авторизации в Яндекс API. Проверьте API ключ")
                        return None
                    elif response.status_code == 403:
                        logger.error("Доступ запрещен. Проверьте права доступа и ID папки")
                        return None
                    elif response.status_code == 429:
                        logger.error("Превышен лимит запросов к API")
                        wait_time = 60 
                        logger.info(f"Ожидание {wait_time} секунд перед следующей попыткой...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Неизвестная ошибка API: {response.text}")
                        return None
                
                translation = response.json()
                logger.debug(f"Получен ответ от API для части {i}")
                
                if 'translations' in translation and translation['translations']:
                    translated_text = translation['translations'][0]['text']
                    translated_parts.append(translated_text)
                    logger.info(f"Часть {i} успешно переведена")
                else:
                    logger.error(f"Неожиданный формат ответа API для части {i}: {translation}")
                    continue
                    
            except requests.exceptions.Timeout:
                logger.error(f"Таймаут при переводе части {i}")
                time.sleep(delay * 2)
                continue
            except requests.exceptions.RequestException as e:
                logger.error(f"Ошибка сети при переводе части {i}: {str(e)}")
                time.sleep(delay)
                continue
            except json.JSONDecodeError as e:
                logger.error(f"Ошибка разбора JSON для части {i}: {str(e)}")
                continue
            except Exception as e:
                logger.error(f"Неожиданная ошибка при переводе части {i}: {str(e)}")
                logger.error(traceback.format_exc())
                continue
            
            # Задержка между запросами для лимитов API
            if i < total_parts:
                logger.debug(f"Ожидание {delay} секунд перед следующим запросом")
                time.sleep(delay)
        
        if not translated_parts:
            logger.error("Не удалось перевести ни одной части текста")
            return None
            
        logger.info(f"Перевод завершен. Успешно переведено {len(translated_parts)} из {total_parts} частей")
        return ' '.join(translated_parts)
        
    except Exception as e:
        logger.error(f"Критическая ошибка при переводе текста: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def main():
    try:
        if not yandexAPI :
            logger.warning("API ключ Яндекс Переводчика не настроен. Функция перевода будет недоступна.")
            
        application = Application.builder().token(telegramAPI).post_init(setup_commands).build()
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("menu", menu))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(CommandHandler("search", search_command))
        application.add_handler(CommandHandler("summary", summary_command))
        application.add_handler(CommandHandler("scientific", scientific_command))
        application.add_handler(CommandHandler("translate", translate_command))
        application.add_handler(CallbackQueryHandler(button_handler))
        application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

        logger.info("Бот запущен")
        application.run_polling(allowed_updates=Update.ALL_TYPES)
    except telegram.error.Conflict as e:
        logger.error("Обнаружен конфликт: другой экземпляр бота уже запущен")
        print("Ошибка: Другой экземпляр бота уже запущен. Пожалуйста, остановите все другие экземпляры бота и попробуйте снова.")
    except Exception as e:
        logger.error(f"Произошла ошибка: {str(e)}")
        print(f"Произошла ошибка: {str(e)}")

if __name__ == '__main__':
    main() 