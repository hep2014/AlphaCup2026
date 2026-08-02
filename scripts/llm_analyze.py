from __future__ import annotations

import argparse
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_ALIASES = {
    "BERT": DEFAULT_MODEL,
    "bert": DEFAULT_MODEL,
    "SBERT": DEFAULT_MODEL,
    "sbert": DEFAULT_MODEL,
}
DEFAULT_TEXT_COLUMN = "text"

REJECTED_COMMENT_TYPES = {
    "promo_spam",
    "gratitude_or_low_value",
    "too_short",
    "consumer_complaint",
    "employee_experience",
    "macro_opinion",
    "other_or_offtopic",
}

COMMENT_METADATA_COLUMNS = (
    "platform",
    "query",
    "source",
    "source_id",
    "post_id",
    "comment_id",
    "parent_id",
    "author_id",
    "author_name",
    "published_at",
    "like_count",
    "url",
    "collected_at",
    "comment_type",
    "inferred_author_type",
)

PAIN_DEFINITIONS = {'unclear_start_steps': {'pain_title': 'Непонятно, с чего начать бизнес',
                         'pain_question': 'Какой первый шаг сделать, какую форму выбрать и в каком '
                                          'порядке всё запускать?',
                         'pain_formula': 'непонимание + первые шаги / открытие бизнеса',
                         'templates': ['не понимаю с чего начать бизнес',
                                       'хочу открыть бизнес, но не знаю первые шаги',
                                       'непонятно как открыть ИП',
                                       'не знаю что выбрать ИП или самозанятость',
                                       'сложно понять порядок запуска бизнеса'],
                         'patterns': ['\\bс чего начать\\b',
                                      '\\bне знаю с чего\\b',
                                      '\\bкак начать\\b',
                                      '\\bкак открыть\\b',
                                      '\\bхочу открыть\\b',
                                      '\\bпланирую открыть\\b',
                                      '\\bдумаю открыть\\b',
                                      '\\bип или самозанят',
                                      '\\bсамозанят\\w* или ип\\b',
                                      '\\bпервые шаги\\b']},
 'fear_of_failure_and_risk': {'pain_title': 'Страшно прогореть и потерять деньги',
                              'pain_question': 'Как снизить риск провала, долгов и личной '
                                               'ответственности?',
                              'pain_formula': 'страх / риск + потеря денег / провал / '
                                              'ответственность',
                              'templates': ['страшно открывать бизнес из-за риска',
                                            'боюсь прогореть и потерять деньги',
                                            'страх провала мешает начать',
                                            'боюсь долгов и ответственности',
                                            'не решаюсь начать из-за риска банкротства'],
                              'patterns': ['\\bбоюсь\\b',
                                           '\\bстрашно\\b',
                                           '\\bстрах\\b',
                                           '\\bриск\\b',
                                           '\\bрисковать\\b',
                                           '\\bпрогор',
                                           '\\bпровал',
                                           '\\bпотерять деньги\\b',
                                           '\\bбанкрот',
                                           '\\bдолг',
                                           '\\bответственност']},
 'lack_of_capital': {'pain_title': 'Не хватает стартового или оборотного капитала',
                     'pain_question': 'Где взять деньги на запуск, закупку, рекламу, найм и рост?',
                     'pain_formula': 'нехватка денег + запуск / закупка / оборотка / рост',
                     'templates': ['нет стартового капитала',
                                   'не хватает денег на запуск бизнеса',
                                   'не хватает оборотных средств',
                                   'нужны деньги на закупку товара',
                                   'нет денег на рекламу и сотрудников'],
                     'patterns': ['\\bнет денег\\b',
                                  '\\bне хватает денег\\b',
                                  '\\bне хватает средств\\b',
                                  '\\bстартов\\w* капитал',
                                  '\\bоборотн\\w* средств',
                                  '\\bоборотк',
                                  '\\bденьги на закуп',
                                  '\\bденьги на реклам',
                                  '\\bденьги на зарплат',
                                  '\\bгде взять деньги\\b',
                                  '\\bкапитал\\b']},
 'credit_and_debt_pressure': {'pain_title': 'Кредиты и долги создают давление',
                              'pain_question': 'Стоит ли брать кредит и как не попасть в долговую '
                                               'ловушку?',
                              'pain_formula': 'кредит / долг + риск / платежи / банкротство',
                              'templates': ['страшно брать кредит на бизнес',
                                            'кредит давит на бизнес',
                                            'долги растут как снежный ком',
                                            'непонятно стоит ли брать кредит',
                                            'банкротство из-за долгов бизнеса'],
                              'patterns': ['\\bкредит\\b',
                                           '\\bзайм\\b',
                                           '\\bдолг',
                                           '\\bплатеж',
                                           '\\bпроцент',
                                           '\\bбанкрот',
                                           '\\bкредитк',
                                           '\\bзанял',
                                           '\\bодолжил']},
 'taxes_reporting_confusion': {'pain_title': 'Непонятны налоги, отчётность и режим',
                               'pain_question': 'Как платить налоги, сдавать отчётность и выбрать '
                                                'ИП/самозанятость/ООО без ошибок?',
                               'pain_formula': 'налоги / отчетность / режим + непонимание / '
                                               'сложность / страх ошибки',
                               'templates': ['не понимаю какие налоги платить',
                                             'сложно разобраться с отчетностью ИП',
                                             'непонятно что выбрать самозанятый или ИП',
                                             'боюсь ошибиться с налогами',
                                             'ФНС начисляет пени и штрафы'],
                               'patterns': ['\\bналог',
                                            '\\bфнс\\b',
                                            '\\bенс\\b',
                                            '\\bип\\b',
                                            '\\bооо\\b',
                                            '\\bсамозанят',
                                            '\\bусн\\b',
                                            '\\bосно\\b',
                                            '\\bндс\\b',
                                            '\\bнпд\\b',
                                            '\\bотчетност',
                                            '\\bдеклараци',
                                            '\\bстрахов\\w* взнос',
                                            '\\bпени\\b',
                                            '\\bвзнос']},
 'legal_fines_blocks': {'pain_title': 'Есть риск штрафов, блокировок и юридических ошибок',
                        'pain_question': 'Как не получить штраф, блокировку счёта, претензии или '
                                         'проблемы с документами?',
                        'pain_formula': 'юридическое требование + штраф / блокировка / ошибка / '
                                        'документы',
                        'templates': ['боюсь штрафов и проверок',
                                      'заблокировали расчетный счет',
                                      'непонятно какие документы нужны',
                                      'сложно разобраться с маркировкой и ЭДО',
                                      'юридические риски бизнеса'],
                        'patterns': ['\\bштраф',
                                     '\\bблокиров',
                                     '\\bарест\\w* счет',
                                     '\\bзакрыли счет',
                                     '\\bпровер',
                                     '\\bдоговор',
                                     '\\bдокумент',
                                     '\\bюрист',
                                     '\\bзакон',
                                     '\\bмаркиров',
                                     '\\bчестн\\w* знак',
                                     '\\bэдо\\b',
                                     '\\bонлайн[- ]?касс',
                                     '\\bсертификат',
                                     '\\bлицензи',
                                     '\\bпретензи',
                                     '\\bроспатент']},
 'no_clients_or_sales': {'pain_title': 'Нет клиентов и первых продаж',
                         'pain_question': 'Как найти клиентов, получить заявки и довести людей до '
                                          'покупки?',
                         'pain_formula': 'нет клиентов / продаж + непонятно как привлекать',
                         'templates': ['нет клиентов',
                                       'не получается найти первых клиентов',
                                       'нет продаж',
                                       'не идут заявки',
                                       'не понимаю как продавать'],
                         'patterns': ['\\bнет клиентов\\b',
                                      '\\bпервые клиенты\\b',
                                      '\\bнайти клиентов\\b',
                                      '\\bпривлечь клиентов\\b',
                                      '\\bнет продаж\\b',
                                      '\\bмало продаж',
                                      '\\bне идут продаж',
                                      '\\bне получается продавать\\b',
                                      '\\bзаявк',
                                      '\\bлид',
                                      '\\bворонк',
                                      '\\bклиент']},
 'marketing_is_expensive_or_unclear': {'pain_title': 'Реклама и продвижение дорогие или непонятные',
                                       'pain_question': 'Как продвигаться, не сливать бюджет и '
                                                        'получать окупаемые заявки?',
                                       'pain_formula': 'реклама / маркетинг + дорого / непонятно / '
                                                       'не окупается',
                                       'templates': ['дорогая реклама не окупается',
                                                     'не понимаю как продвигать бизнес',
                                                     'сливаю бюджет на рекламу',
                                                     'нет результата от маркетинга',
                                                     'непонятно где искать клиентов'],
                                       'patterns': ['\\bреклам',
                                                    '\\bмаркетинг',
                                                    '\\bпродвиж',
                                                    '\\bтаргет',
                                                    '\\bseo\\b',
                                                    '\\bsmm\\b',
                                                    '\\bконтент',
                                                    '\\bподписчик',
                                                    '\\bбюджет',
                                                    '\\bокупа',
                                                    '\\bслив',
                                                    '\\bдорог']},
 'weak_margin_or_unit_economics': {'pain_title': 'Маржа и экономика не сходятся',
                                   'pain_question': 'Как продавать с прибылью, если расходы, '
                                                    'комиссии и себестоимость съедают доход?',
                                   'pain_formula': 'маржа / прибыль / себестоимость + не сходится '
                                                   '/ минус / съедают расходы',
                                   'templates': ['маржа слишком маленькая',
                                                 'экономика не сходится',
                                                 'работаю в минус',
                                                 'комиссии съедают прибыль',
                                                 'не понимаю как посчитать окупаемость'],
                                   'patterns': ['\\bмарж',
                                                '\\bприбыл',
                                                '\\bубыт',
                                                '\\bв минус\\b',
                                                '\\bминус',
                                                '\\bсебестоим',
                                                '\\bэкономик',
                                                '\\bюнит',
                                                '\\bокупаемост',
                                                '\\bрентабельн',
                                                '\\bрасход',
                                                '\\bзатрат',
                                                '\\bсъеда',
                                                '\\bне сход',
                                                '\\bне окупа']},
 'marketplace_commissions_rules': {'pain_title': 'Маркетплейсы забирают прибыль комиссиями и '
                                                 'правилами',
                                   'pain_question': 'Как выжить селлеру при комиссиях, штрафах, '
                                                    'логистике, хранении, возвратах и изменении '
                                                    'правил?',
                                   'pain_formula': 'маркетплейс + комиссия / штраф / логистика / '
                                                   'возврат / правила + убыток',
                                   'templates': ['маркетплейс забирает большую комиссию',
                                                 'комиссии и логистика съедают прибыль',
                                                 'штрафы на озоне и вайлдберриз',
                                                 'возвраты и подмены товара приводят к убыткам',
                                                 'постоянно меняются правила маркетплейса'],
                                   'patterns': ['\\bмаркетплейс',
                                                '\\bozon\\b',
                                                '\\bозон\\b',
                                                '\\bwildberries\\b',
                                                '\\bвайлдберриз\\b',
                                                '\\bвб\\b',
                                                '\\bяндекс маркет\\b',
                                                '\\bселлер',
                                                '\\bкомисси',
                                                '\\bспп\\b',
                                                '\\bлогистик',
                                                '\\bхранени',
                                                '\\bвозврат',
                                                '\\bподмен',
                                                '\\bштраф',
                                                '\\bфулфилмент',
                                                '\\bоферт',
                                                '\\bвыкуп',
                                                '\\bсклад']},
 'product_niche_supplier_uncertainty': {'pain_title': 'Сложно выбрать товар, нишу или поставщика',
                                        'pain_question': 'Как найти товар, нишу, поставщика и '
                                                         'цену, при которых будет спрос и прибыль?',
                                        'pain_formula': 'товар / ниша / поставщик + '
                                                        'неопределенность / спрос / цена',
                                        'templates': ['не знаю какой товар продавать',
                                                      'сложно выбрать нишу',
                                                      'непонятно где найти поставщика',
                                                      'непонятен спрос на товар',
                                                      'цена и себестоимость не дают заработать'],
                                        'patterns': ['\\bниш',
                                                     '\\bтовар',
                                                     '\\bпоставщик',
                                                     '\\bзакуп',
                                                     '\\bспрос',
                                                     '\\bассортимент',
                                                     '\\bцена\\b',
                                                     '\\bсебестоим',
                                                     '\\bпроизвод',
                                                     '\\bбренд',
                                                     '\\bостатк',
                                                     '\\bкачество']},
 'banking_payments_acquiring_friction': {'pain_title': 'Банки, платежи и эквайринг создают трение',
                                         'pain_question': 'Как выбрать банк, принимать платежи и '
                                                          'не потерять деньги на комиссиях или '
                                                          'блокировках?',
                                         'pain_formula': 'банк / счет / эквайринг + комиссия / '
                                                         'блокировка / выбор',
                                         'templates': ['какой банк выбрать для ИП',
                                                       'эквайринг дорогой',
                                                       'банк заблокировал счет',
                                                       'непонятно как принимать платежи',
                                                       'комиссия банка съедает прибыль'],
                                         'patterns': ['\\bбанк',
                                                      '\\bрасчетн\\w* счет',
                                                      '\\bрасч[её]тн\\w* счет',
                                                      '\\bэквайринг',
                                                      '\\bплатеж',
                                                      '\\bперевод',
                                                      '\\bтерминал',
                                                      '\\bкомисси\\w* банк',
                                                      '\\bкарта\\b',
                                                      '\\bтинькофф\\b',
                                                      '\\bт[- ]?банк\\b',
                                                      '\\bсбер\\b']},
 'hiring_delegation_blocker': {'pain_title': 'Не получается нанять, делегировать и выйти из '
                                             'операционки',
                               'pain_question': 'Как нанимать людей, платить им и передавать '
                                                'задачи без потери контроля и денег?',
                               'pain_formula': 'найм / сотрудники / делегирование + не хватает '
                                               'денег / контроля / времени',
                               'templates': ['не хватает денег на сотрудников',
                                             'нужно нанимать людей, но страшно',
                                             'не получается делегировать',
                                             'всё делаю сам и не могу вырасти',
                                             'операционка мешает развивать бизнес'],
                               'patterns': ['\\bнаним',
                                            '\\bнайм',
                                            '\\bсотрудник',
                                            '\\bкоманд',
                                            '\\bагент',
                                            '\\bделегир',
                                            '\\bоперацион',
                                            '\\bвсе делаю сам',
                                            '\\bсам продаю\\b',
                                            '\\bзарплат',
                                            '\\bуправля',
                                            '\\bлюдей\\b']},
 'overload_burnout_time': {'pain_title': 'Не хватает времени, много рутины и перегруз',
                           'pain_question': 'Как не утонуть в рутине, не выгореть и освободить '
                                            'время на развитие?',
                           'pain_formula': 'не хватает времени / рутина / стресс / усталость',
                           'templates': ['работаю без выходных',
                                         'не хватает времени на бизнес',
                                         'устал от рутины',
                                         'выгорание',
                                         'не успеваю заниматься всем'],
                           'patterns': ['\\bне хватает времени\\b',
                                        '\\bнет времени\\b',
                                        '\\bустал',
                                        '\\bвыгоран',
                                        '\\bбез выходных\\b',
                                        '\\bрутин',
                                        '\\bне успева',
                                        '\\bстресс',
                                        '\\bнерв',
                                        '\\bперегруз']},
 'scaling_growth_blocker': {'pain_title': 'Непонятно, как расти и масштабироваться',
                            'pain_question': 'Как увеличить продажи, выйти на новый уровень и не '
                                             'сломать процессы?',
                            'pain_formula': 'рост / масштабирование + неясность / нехватка '
                                            'ресурсов / потеря качества',
                            'templates': ['не понимаю как масштабировать бизнес',
                                          'непонятно куда расти дальше',
                                          'хочу увеличить продажи, но не получается',
                                          'открыл точки и упало качество',
                                          'не хватает ресурсов на рост'],
                            'patterns': ['\\bмасштаб',
                                         '\\bрост\\b',
                                         '\\bрасти\\b',
                                         '\\bувеличить продаж',
                                         '\\bувеличить оборот',
                                         '\\bкуда двигаться\\b',
                                         '\\bчто дальше\\b',
                                         '\\bновый уровень\\b',
                                         '\\bнесколько точек\\b',
                                         '\\bразвивать бизнес\\b']}}

BUSINESS_CONTEXT_PATTERNS = ['\\bбизнес\\b',
 '\\bпредпринимател',
 '\\bип\\b',
 '\\bооо\\b',
 '\\bсамозанят',
 '\\bналог',
 '\\bфнс\\b',
 '\\bбанк\\b',
 '\\bкредит\\b',
 '\\bклиент',
 '\\bпродаж',
 '\\bмаркетинг\\b',
 '\\bреклам',
 '\\bмаркетплейс',
 '\\bozon\\b',
 '\\bозон\\b',
 '\\bwildberries\\b',
 '\\bвайлдберриз\\b',
 '\\bвб\\b',
 '\\bселлер',
 '\\bпоставщик',
 '\\bтовар',
 '\\bниш',
 '\\bмарж',
 '\\bприбыл',
 '\\bвыручк',
 '\\bоборот',
 '\\bаренд',
 '\\bкасс',
 '\\bбухгалтер',
 '\\bотчет',
 '\\bсотрудник',
 '\\bкоманд',
 '\\bзакуп',
 '\\bпоставк']

PAIN_SIGNAL_PATTERNS = ['\\bне понимаю\\b',
 '\\bнепонятн',
 '\\bне знаю\\b',
 '\\bсложно\\b',
 '\\bтрудно\\b',
 '\\bбоюсь\\b',
 '\\bстрашно\\b',
 '\\bстрах\\b',
 '\\bпроблем',
 '\\bне получается\\b',
 '\\bне выходит\\b',
 '\\bмешает\\b',
 '\\bнет\\b',
 '\\bне хватает\\b',
 '\\bдорого\\b',
 '\\bсъеда',
 '\\bв минус\\b',
 '\\bубыт',
 '\\bриск',
 '\\bштраф',
 '\\bпени\\b',
 '\\bблокиров',
 '\\bдолг',
 '\\bбанкрот',
 '\\bпрогор',
 '\\bпотер',
 '\\bзакрыли\\b',
 '\\bне дают\\b',
 '\\bне работает\\b',
 '\\bне окупа',
 '\\bне сход',
 '\\bнужно\\b',
 '\\bнадо\\b',
 '\\bкак\\b',
 '\\bчто делать\\b',
 '\\bстоит ли\\b',
 '\\bможно ли\\b',
 '\\bгде взять\\b',
 '\\bкак найти\\b']

QUESTION_PATTERNS = ['\\bкак\\b',
 '\\bчто делать\\b',
 '\\bкуда\\b',
 '\\bгде\\b',
 '\\bпочему\\b',
 '\\bзачем\\b',
 '\\bсколько\\b',
 '\\bстоит ли\\b',
 '\\bможно ли\\b',
 '\\bкакой\\b',
 '\\bкакую\\b',
 '\\bкакие\\b']

CONSUMER_ONLY_PATTERNS = ['\\bкак покупател',
 '\\bмнение как покупател',
 '\\bя покупател',
 '\\bкупил[аи]?\\b',
 '\\bзаказал[аи]?\\b',
 '\\bпокупал[аи]?\\b',
 '\\bпосылк',
 '\\bдоставк',
 '\\bприш[её]л товар',
 '\\bденьги не вернули',
 '\\bвозврат денег\\b',
 '\\bбракованн',
 '\\bгаранти']

ENTREPRENEUR_MARKERS = ['\\bя ип\\b',
 '\\bу меня ип\\b',
 '\\bу меня ооо\\b',
 '\\bя предпринимател',
 '\\bмой бизнес\\b',
 '\\bмой магазин\\b',
 '\\bу меня магазин\\b',
 '\\bя производитель\\b',
 '\\bя селлер\\b',
 '\\bпродаю\\b',
 '\\bмы продаем\\b',
 '\\bторгую\\b',
 '\\bмои клиенты\\b',
 '\\bмои продажи\\b',
 '\\bу моего товара\\b',
 '\\bс моего товара\\b',
 '\\bработал с озон']

PROMO_PATTERNS = ['\\bподписывай',
 '\\bзабирай',
 '\\bмини[- ]?курс',
 '\\bрегистрация',
 '\\bбесплатно\\b',
 '\\bконсультац',
 '\\bпромокод',
 '\\bt\\.me\\b',
 '\\btelegram\\b',
 '\\bтелеграм\\b',
 '\\bhttp',
 '\\bссылка\\b']

STOPWORDS = {'а',
 'автор',
 'бизнес',
 'бы',
 'в',
 'вам',
 'вас',
 'видео',
 'во',
 'все',
 'всё',
 'вы',
 'где',
 'да',
 'дело',
 'для',
 'до',
 'его',
 'ее',
 'если',
 'еще',
 'ещё',
 'её',
 'же',
 'за',
 'и',
 'из',
 'или',
 'их',
 'к',
 'как',
 'канал',
 'когда',
 'которая',
 'которое',
 'которые',
 'который',
 'ли',
 'меня',
 'мне',
 'мое',
 'можно',
 'мой',
 'моя',
 'мы',
 'на',
 'надо',
 'нам',
 'нас',
 'не',
 'нет',
 'ни',
 'но',
 'нужно',
 'о',
 'об',
 'он',
 'она',
 'они',
 'от',
 'очень',
 'по',
 'просто',
 'работа',
 'работать',
 'с',
 'сам',
 'сама',
 'свое',
 'свой',
 'своя',
 'со',
 'так',
 'там',
 'то',
 'тоже',
 'только',
 'тут',
 'у',
 'уже',
 'что',
 'чтобы',
 'это',
 'я'}

@dataclass(frozen=True, slots=True)
class PainDefinition:
    """Описание одной бизнес-боли и правил её распознавания."""

    pain_id: str
    title: str
    question: str
    formula: str
    templates: tuple[str, ...]
    patterns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """Все изменяемые параметры одного запуска анализа."""

    input_path: Path
    output_dir: Path
    model_name: str = DEFAULT_MODEL
    semantic_threshold: float = 0.36
    combined_threshold: float = 0.34
    max_labels: int = 3
    use_existing_filter: bool = True
    text_column: str = DEFAULT_TEXT_COLUMN

    def validate(self) -> None:
        if not self.input_path.exists():
            raise FileNotFoundError(f"Входной файл не найден: {self.input_path}")
        if not 0.0 <= self.semantic_threshold <= 1.0:
            raise ValueError("semantic_threshold должен находиться в диапазоне [0, 1]")
        if not 0.0 <= self.combined_threshold <= 1.0:
            raise ValueError("combined_threshold должен находиться в диапазоне [0, 1]")
        if self.max_labels < 1:
            raise ValueError("max_labels должен быть положительным числом")


@dataclass(slots=True)
class PreparedComments:
    """Результат первичной очистки и статистика отсева строк."""

    dataframe: pd.DataFrame
    statistics: dict[str, int]


@dataclass(slots=True)
class ExtractedPainUnits:
    """Фрагменты, принятые и отклонённые эвристическим фильтром."""

    accepted: pd.DataFrame
    rejected: pd.DataFrame


@dataclass(slots=True)
class AnalysisArtifacts:
    """Все таблицы и текстовый отчёт, полученные конвейером."""

    comments: pd.DataFrame
    pain_units: pd.DataFrame
    rejected_fragments: pd.DataFrame
    comments_with_pains: pd.DataFrame
    pain_summary: pd.DataFrame
    query_summary: pd.DataFrame
    report: str
    preparation_statistics: dict[str, int]


@dataclass(frozen=True, slots=True)
class OutputPaths:
    """Имена выходных файлов собраны в одном месте."""

    comments: Path
    pain_units: Path
    rejected_fragments: Path
    pain_summary: Path
    query_summary: Path
    report: Path
    audit: Path

    @classmethod
    def inside(cls, directory: Path) -> "OutputPaths":
        return cls(
            comments=directory / "comments_with_extracted_pains.csv",
            pain_units=directory / "pain_units.csv",
            rejected_fragments=directory / "rejected_fragments.csv",
            pain_summary=directory / "pain_summary.csv",
            query_summary=directory / "query_pain_summary.csv",
            report=directory / "report_v3.md",
            audit=directory / "pain_analysis_audit.xlsx",
        )


class TableReader(Protocol):
    """Абстракция источника табличных данных."""

    def read(self, path: Path) -> pd.DataFrame:
        ...


class TextEncoder(Protocol):
    """Минимальный интерфейс модели, необходимый классификатору."""

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> np.ndarray:
        ...


class TextEncoderFactory(Protocol):
    """Создаёт модель по имени, не связывая конвейер с конкретной библиотекой."""

    def create(self, model_name: str) -> TextEncoder:
        ...


class ResultsExporter(Protocol):
    """Сохраняет итоговые артефакты анализа."""

    def export(self, artifacts: AnalysisArtifacts, output_dir: Path) -> OutputPaths:
        ...


class PandasTableReader:
    """Читает CSV и Excel, скрывая детали кодировок от бизнес-логики."""

    def read(self, path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path)

        try:
            return pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            LOGGER.warning("UTF-8 не подошёл, пробую прочитать CSV в кодировке cp1251")
            return pd.read_csv(path, encoding="cp1251")


class SentenceTransformerEncoderFactory:
    """Ленивая фабрика SentenceTransformer.

    Импорт выполняется только перед классификацией. Благодаря этому команда
    ``--help`` и тесты остальных компонентов не требуют загрузки ML-библиотеки.
    """

    def create(self, model_name: str) -> TextEncoder:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Для классификации установите пакет sentence-transformers"
            ) from exc

        LOGGER.info("Загружаю BERT/SBERT-модель: %s", model_name)
        return SentenceTransformer(model_name)


class TextTools:
    """Чистые функции очистки текста без скрытого состояния."""

    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+|t\.me/\S+", re.IGNORECASE)
    WORD_PATTERN = re.compile(r"[а-яa-z0-9]+", re.IGNORECASE)
    LEMMA_WORD_PATTERN = re.compile(r"[а-яa-zA-Z\-]{3,}")

    @staticmethod
    def safe_string(value: Any) -> str:
        return "" if pd.isna(value) else str(value)

    @classmethod
    def normalize_for_rules(cls, text: Any) -> str:
        normalized = cls.safe_string(text).lower().replace("ё", "е")
        normalized = cls.URL_PATTERN.sub(" ", normalized)
        normalized = normalized.replace("\n", " ").replace("\r", " ")
        return re.sub(r"\s+", " ", normalized).strip()

    @classmethod
    def clean_for_embedding(cls, text: Any) -> str:
        normalized = cls.safe_string(text).replace("ё", "е")
        normalized = cls.URL_PATTERN.sub(" ", normalized)
        normalized = re.sub(r"@\w+", " ", normalized)
        normalized = re.sub(r"#\w+", " ", normalized)
        normalized = normalized.replace("\n", " ").replace("\r", " ")
        return re.sub(r"\s+", " ", normalized).strip()

    @classmethod
    def count_urls(cls, text: Any) -> int:
        return len(cls.URL_PATTERN.findall(cls.safe_string(text).lower()))

    @classmethod
    def word_count(cls, text: Any) -> int:
        return len(cls.WORD_PATTERN.findall(cls.normalize_for_rules(text)))

    @classmethod
    def truncate(cls, text: Any, limit: int = 850) -> str:
        normalized = re.sub(
            r"\s+",
            " ",
            cls.safe_string(text).replace("\n", " "),
        ).strip()
        if len(normalized) <= limit:
            return normalized
        return normalized[:limit].rstrip() + "..."

    @staticmethod
    def bool_value(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return TextTools.safe_string(value).strip().lower() in {
            "true",
            "1",
            "yes",
            "да",
            "y",
        }

    @staticmethod
    def clean_pattern(pattern: str) -> str:
        cleaned = (
            pattern.replace(r"\b", "")
            .replace(r"\w*", "")
            .replace("[а-я]*", "")
            .replace("\\", "")
        )
        return cleaned.strip("^$")[:70]


class CompiledPatterns:
    """Набор заранее скомпилированных регулярных выражений."""

    def __init__(self, patterns: Iterable[str]) -> None:
        self._patterns = tuple(
            (pattern, re.compile(pattern, re.IGNORECASE)) for pattern in patterns
        )

    def hits(self, normalized_text: str) -> list[str]:
        return [raw for raw, compiled in self._patterns if compiled.search(normalized_text)]


class PainCatalog:
    """Хранилище определений болей и связанных с ними правил."""

    def __init__(self, definitions: Mapping[str, Mapping[str, Any]]) -> None:
        self._definitions = {
            pain_id: PainDefinition(
                pain_id=pain_id,
                title=str(data["pain_title"]),
                question=str(data["pain_question"]),
                formula=str(data["pain_formula"]),
                templates=tuple(data["templates"]),
                patterns=tuple(data["patterns"]),
            )
            for pain_id, data in definitions.items()
        }
        self._matchers = {
            pain_id: CompiledPatterns(definition.patterns)
            for pain_id, definition in self._definitions.items()
        }

    def __iter__(self):
        return iter(self._definitions.values())

    def __len__(self) -> int:
        return len(self._definitions)

    def get(self, pain_id: str) -> PainDefinition:
        return self._definitions[pain_id]

    def ids(self) -> list[str]:
        return list(self._definitions)

    def rule_hits(self, pain_id: str, normalized_text: str) -> list[str]:
        return self._matchers[pain_id].hits(normalized_text)

    def embedding_descriptions(self) -> list[str]:
        descriptions = []
        for definition in self:
            descriptions.append(
                f"Боль: {definition.title}. "
                f"Вопрос пользователя: {definition.question} "
                f"Формула боли: {definition.formula}. "
                "Типичные формулировки: "
                + ". ".join(definition.templates)
            )
        return descriptions


class CommentPreprocessor:
    """Очищает, фильтрует и дедуплицирует исходные комментарии."""

    def __init__(self, text_column: str = DEFAULT_TEXT_COLUMN) -> None:
        self._text_column = text_column

    def prepare(
        self,
        dataframe: pd.DataFrame,
        *,
        use_existing_filter: bool,
    ) -> PreparedComments:
        if self._text_column not in dataframe.columns:
            raise ValueError(
                f"Нет колонки '{self._text_column}'. "
                f"Колонки файла: {list(dataframe.columns)}"
            )

        comments = dataframe.copy()
        statistics = {"input_rows": len(comments)}
        comments["raw_text"] = comments[self._text_column].fillna("").astype(str)
        comments["clean_text"] = comments["raw_text"].map(TextTools.clean_for_embedding)
        comments["word_count"] = comments["raw_text"].map(TextTools.word_count)
        comments["url_count"] = comments["raw_text"].map(TextTools.count_urls)

        if use_existing_filter and "suitable_for_pain_analysis" in comments.columns:
            comments = self._apply_filter(
                comments,
                comments["suitable_for_pain_analysis"].map(TextTools.bool_value),
                statistics,
                result_key="after_existing_suitability",
                removed_key="removed_by_existing_suitability",
            )

        if use_existing_filter and "comment_type" in comments.columns:
            accepted_types = ~comments["comment_type"].fillna("").isin(
                REJECTED_COMMENT_TYPES
            )
            comments = self._apply_filter(
                comments,
                accepted_types,
                statistics,
                result_key="after_comment_type_filter",
                removed_key="removed_by_comment_type_filter",
            )

        basic_mask = (comments["word_count"] >= 6) & (comments["url_count"] < 3)
        comments = self._apply_filter(
            comments,
            basic_mask,
            statistics,
            result_key="after_basic_filter",
            removed_key="removed_by_basic_filter",
        )

        before_deduplication = len(comments)
        comments["dedup_key"] = comments["clean_text"].str.lower().str.strip()
        comments = comments[comments["dedup_key"] != ""].drop_duplicates(
            subset=["dedup_key"]
        )
        statistics["after_dedup"] = len(comments)
        statistics["removed_by_dedup"] = before_deduplication - len(comments)

        comments = comments.reset_index(drop=True)
        comments["comment_row_id"] = np.arange(len(comments), dtype=int)
        return PreparedComments(comments, statistics)

    @staticmethod
    def _apply_filter(
        dataframe: pd.DataFrame,
        mask: pd.Series,
        statistics: dict[str, int],
        *,
        result_key: str,
        removed_key: str,
    ) -> pd.DataFrame:
        before = len(dataframe)
        filtered = dataframe.loc[mask].copy()
        statistics[result_key] = len(filtered)
        statistics[removed_key] = before - len(filtered)
        return filtered


class FragmentSplitter:
    """Разбивает длинный комментарий на самостоятельные смысловые фрагменты."""

    SHORT_FRAGMENT_WORDS = 5
    CLAUSE_SEPARATOR = re.compile(
        r";\s+|,\s+(?=(?:но|а|если|когда|потому|так как|при этом|и еще|ещё)\b)",
        re.IGNORECASE,
    )
    SENTENCE_SEPARATOR = re.compile(r"(?<=[.!?…])\s+|[\n\r]+")
    CONNECTING_WORDS = {
        "но",
        "а",
        "если",
        "когда",
        "потому",
        "так как",
        "при этом",
        "и еще",
        "ещё",
    }

    def __init__(self, max_length: int = 420) -> None:
        self._max_length = max_length

    def split(self, text: Any) -> list[str]:
        normalized = re.sub(r"\s+", " ", TextTools.safe_string(text)).strip()
        if not normalized:
            return []

        fragments: list[str] = []
        for sentence in self.SENTENCE_SEPARATOR.split(normalized):
            sentence = sentence.strip(" -–—\t")
            if not sentence:
                continue
            if len(sentence) <= self._max_length:
                fragments.append(sentence)
                continue
            fragments.extend(self._split_long_sentence(sentence))

        return self._merge_short_fragments(fragments)

    def _split_long_sentence(self, sentence: str) -> list[str]:
        result: list[str] = []
        buffer = ""

        for clause in self.CLAUSE_SEPARATOR.split(sentence):
            clause = clause.strip(" -–—,\t")
            if not clause or clause.lower() in self.CONNECTING_WORDS:
                continue

            candidate = f"{buffer} {clause}".strip()
            if len(candidate) < self._max_length:
                buffer = candidate
                continue

            if buffer:
                result.append(buffer)
            buffer = clause

        if buffer:
            result.append(buffer)
        return result

    def _merge_short_fragments(self, fragments: Sequence[str]) -> list[str]:
        merged: list[str] = []
        buffer = ""

        for fragment in fragments:
            if TextTools.word_count(fragment) < self.SHORT_FRAGMENT_WORDS:
                buffer = f"{buffer} {fragment}".strip()
                continue

            if buffer:
                fragment = f"{buffer} {fragment}".strip()
                buffer = ""
            merged.append(fragment)

        if buffer and merged:
            merged[-1] = f"{merged[-1]} {buffer}".strip()
        elif buffer:
            merged.append(buffer)

        return [
            fragment
            for fragment in merged
            if TextTools.word_count(fragment) >= self.SHORT_FRAGMENT_WORDS
        ]


class PainLikelihoodEvaluator:
    """Определяет, похож ли фрагмент на содержательное описание боли."""

    def __init__(self) -> None:
        self._business = CompiledPatterns(BUSINESS_CONTEXT_PATTERNS)
        self._pain_signals = CompiledPatterns(PAIN_SIGNAL_PATTERNS)
        self._questions = CompiledPatterns(QUESTION_PATTERNS)
        self._consumer_only = CompiledPatterns(CONSUMER_ONLY_PATTERNS)
        self._entrepreneur = CompiledPatterns(ENTREPRENEUR_MARKERS)
        self._promo = CompiledPatterns(PROMO_PATTERNS)

    def evaluate(self, fragment: str, full_comment: str = "") -> dict[str, Any]:
        normalized_fragment = TextTools.normalize_for_rules(fragment)
        normalized_context = TextTools.normalize_for_rules(
            f"{fragment} {full_comment[:500]}"
        )

        business_hits = self._business.hits(normalized_context)
        pain_hits = self._pain_signals.hits(normalized_fragment)
        question_hits = self._questions.hits(normalized_fragment)
        consumer_hits = self._consumer_only.hits(normalized_fragment)
        entrepreneur_hits = self._entrepreneur.hits(normalized_context)
        promo_hits = self._promo.hits(normalized_fragment)

        word_count = TextTools.word_count(fragment)
        is_question = "?" in fragment or bool(question_hits)
        has_business_context = bool(business_hits)
        has_pain_signal = bool(pain_hits)
        has_entrepreneur_marker = bool(entrepreneur_hits)
        consumer_only = bool(consumer_hits) and not has_entrepreneur_marker

        score = min(len(business_hits), 4) * 1.0
        score += min(len(pain_hits), 5) * 1.8
        score += 1.2 if is_question else 0.0
        score += 1.5 if has_entrepreneur_marker else 0.0
        score -= 3.5 if consumer_only else 0.0
        score -= 4.0 if len(promo_hits) >= 2 else 0.0
        score -= 2.0 if word_count < 7 else 0.0

        should_keep = (
            word_count >= 7
            and has_business_context
            and (has_pain_signal or is_question)
            and len(promo_hits) < 2
            and score >= 4.0
        )
        if (
            has_entrepreneur_marker
            and has_business_context
            and (has_pain_signal or is_question)
            and score >= 3.5
        ):
            should_keep = True

        return {
            "is_pain_unit": should_keep,
            "pain_likelihood_score": round(score, 3),
            "business_hits": self._format_hits(business_hits, 8),
            "pain_signal_hits": self._format_hits(pain_hits, 8),
            "question_hits": self._format_hits(question_hits, 5),
            "entrepreneur_hits": self._format_hits(entrepreneur_hits, 5),
            "consumer_hits": self._format_hits(consumer_hits, 5),
        }

    @staticmethod
    def _format_hits(hits: Sequence[str], limit: int) -> str:
        return "; ".join(TextTools.clean_pattern(pattern) for pattern in hits[:limit])


class PainUnitExtractor:
    """Извлекает из комментариев фрагменты, пригодные для классификации."""

    def __init__(
        self,
        splitter: FragmentSplitter,
        evaluator: PainLikelihoodEvaluator,
    ) -> None:
        self._splitter = splitter
        self._evaluator = evaluator

    def extract(self, comments: pd.DataFrame) -> ExtractedPainUnits:
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []

        for _, row in comments.iterrows():
            full_text = row["raw_text"]
            for fragment_id, fragment in enumerate(self._splitter.split(full_text)):
                base_record: dict[str, Any] = {
                    "comment_row_id": row["comment_row_id"],
                    "fragment_id": fragment_id,
                    "pain_text": fragment,
                    "source_comment": full_text,
                }
                for column in COMMENT_METADATA_COLUMNS:
                    if column in row.index:
                        base_record[column] = row[column]

                evaluation = self._evaluator.evaluate(fragment, full_text)
                record = {**base_record, **evaluation}
                target = accepted if evaluation["is_pain_unit"] else rejected
                target.append(record)

        accepted_frame = pd.DataFrame(accepted)
        rejected_frame = pd.DataFrame(rejected)

        if not accepted_frame.empty:
            accepted_frame = accepted_frame.reset_index(drop=True)
            accepted_frame["pain_unit_id"] = np.arange(len(accepted_frame), dtype=int)
            accepted_frame["clean_pain_text"] = accepted_frame["pain_text"].map(
                TextTools.clean_for_embedding
            )

        return ExtractedPainUnits(accepted_frame, rejected_frame)


class HybridPainClassifier:
    """Сочетает семантическое сходство эмбеддингов и regex-правила."""

    SEMANTIC_WEIGHT = 0.76
    RULE_WEIGHT = 0.24

    def __init__(self, catalog: PainCatalog) -> None:
        self._catalog = catalog

    def classify(
        self,
        pain_units: pd.DataFrame,
        encoder: TextEncoder,
        *,
        semantic_threshold: float,
        combined_threshold: float,
        max_labels: int,
    ) -> pd.DataFrame:
        if pain_units.empty:
            return pain_units.copy()

        units = pain_units.reset_index(drop=True).copy()
        texts = units["clean_pain_text"].fillna("").tolist()

        LOGGER.info("Считаю эмбеддинги %s фрагментов", len(texts))
        unit_embeddings = np.asarray(
            encoder.encode(
                texts,
                batch_size=32,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
        )
        definition_embeddings = np.asarray(
            encoder.encode(
                self._catalog.embedding_descriptions(),
                batch_size=16,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )

        labels = self._catalog.ids()
        semantic_scores = unit_embeddings @ definition_embeddings.T
        classification_rows: list[dict[str, Any]] = []

        for position, row in units.iterrows():
            normalized_text = TextTools.normalize_for_rules(row["pain_text"])
            rule_scores, rule_hits = self._calculate_rule_scores(normalized_text)
            raw_rules = np.array([rule_scores[label] for label in labels], dtype=float)
            normalized_rules = np.clip(raw_rules / 3.0, 0.0, 1.0)
            combined_scores = (
                self.SEMANTIC_WEIGHT * semantic_scores[position]
                + self.RULE_WEIGHT * normalized_rules
            )
            ranking = np.argsort(combined_scores)[::-1]

            assigned, titles, details = self._select_labels(
                ranking=ranking,
                labels=labels,
                semantic_scores=semantic_scores[position],
                raw_rule_scores=raw_rules,
                combined_scores=combined_scores,
                rule_hits=rule_hits,
                semantic_threshold=semantic_threshold,
                combined_threshold=combined_threshold,
                max_labels=max_labels,
            )

            if not assigned:
                best_index = int(ranking[0])
                best_label = labels[best_index]
                assigned = [best_label]
                titles = [self._catalog.get(best_label).title]
                details = [
                    f"{best_label}: semantic={semantic_scores[position, best_index]:.3f}, "
                    f"rule={int(raw_rules[best_index])}, "
                    f"combined={combined_scores[best_index]:.3f}, weak_best=True"
                ]

            primary_label = assigned[0]
            primary_index = labels.index(primary_label)
            primary = self._catalog.get(primary_label)
            classification_rows.append(
                {
                    "primary_pain_id": primary_label,
                    "primary_pain_title": primary.title,
                    "primary_pain_question": primary.question,
                    "primary_pain_formula": primary.formula,
                    "primary_semantic_score": round(
                        float(semantic_scores[position, primary_index]), 4
                    ),
                    "primary_rule_score": int(raw_rules[primary_index]),
                    "primary_combined_score": round(
                        float(combined_scores[primary_index]), 4
                    ),
                    "pain_ids": "; ".join(assigned),
                    "pain_titles": "; ".join(titles),
                    "classification_details": " | ".join(details),
                    "top_5_scores": " | ".join(
                        f"{labels[index]}={combined_scores[index]:.3f}"
                        for index in ranking[:5]
                    ),
                }
            )

        return pd.concat(
            [units, pd.DataFrame(classification_rows)],
            axis=1,
        )

    def _calculate_rule_scores(
        self,
        normalized_text: str,
    ) -> tuple[dict[str, int], dict[str, list[str]]]:
        scores: dict[str, int] = {}
        hits: dict[str, list[str]] = {}
        for pain_id in self._catalog.ids():
            pain_hits = self._catalog.rule_hits(pain_id, normalized_text)
            scores[pain_id] = len(pain_hits)
            hits[pain_id] = pain_hits
        return scores, hits

    def _select_labels(
        self,
        *,
        ranking: np.ndarray,
        labels: Sequence[str],
        semantic_scores: np.ndarray,
        raw_rule_scores: np.ndarray,
        combined_scores: np.ndarray,
        rule_hits: Mapping[str, Sequence[str]],
        semantic_threshold: float,
        combined_threshold: float,
        max_labels: int,
    ) -> tuple[list[str], list[str], list[str]]:
        assigned: list[str] = []
        titles: list[str] = []
        details: list[str] = []

        for index in ranking:
            label = labels[int(index)]
            semantic = float(semantic_scores[index])
            rule_score = int(raw_rule_scores[index])
            combined = float(combined_scores[index])
            passed = (
                combined >= combined_threshold
                or (semantic >= semantic_threshold and rule_score >= 1)
                or rule_score >= 2
            )
            if passed:
                assigned.append(label)
                titles.append(self._catalog.get(label).title)
                cleaned_hits = "; ".join(
                    TextTools.clean_pattern(pattern)
                    for pattern in rule_hits[label][:5]
                )
                details.append(
                    f"{label}: semantic={semantic:.3f}, rule={rule_score}, "
                    f"combined={combined:.3f}, hits={cleaned_hits}"
                )
            if len(assigned) >= max_labels:
                break

        return assigned, titles, details


class RussianLemmatizer:
    """Лемматизатор с безопасным откатом при отсутствии pymorphy3."""

    def __init__(self) -> None:
        self._lemmatize = self._build_lemmatizer()

    def __call__(self, word: str) -> str:
        return self._lemmatize(word)

    @staticmethod
    def _build_lemmatizer() -> Callable[[str], str]:
        try:
            import pymorphy3

            morph = pymorphy3.MorphAnalyzer()
            LOGGER.info("Для сводок используется лемматизация pymorphy3")
            return lambda word: morph.parse(word)[0].normal_form.replace("ё", "е")
        except Exception as exc:
            LOGGER.warning(
                "pymorphy3 недоступен, слова останутся без лемматизации: %s",
                exc,
            )
            return lambda word: word.replace("ё", "е")


class SummaryBuilder:
    """Строит сводки на уровне болей, комментариев и поисковых запросов."""

    def __init__(self, catalog: PainCatalog, lemmatizer: RussianLemmatizer) -> None:
        self._catalog = catalog
        self._lemmatizer = lemmatizer

    def build_pain_summary(
        self,
        units: pd.DataFrame,
        *,
        total_comments: int,
    ) -> pd.DataFrame:
        if units.empty:
            return pd.DataFrame()

        units_with_lemmas = self._add_lemmas(units, "pain_text")
        rows: list[dict[str, Any]] = []

        for definition in self._catalog:
            mask = units_with_lemmas["pain_ids"].fillna("").map(
                lambda value: definition.pain_id in self._split_labels(value)
            )
            part = units_with_lemmas.loc[mask].copy()
            rows.append(
                self._build_pain_summary_row(
                    definition,
                    part,
                    total_comments=total_comments,
                )
            )

        return (
            pd.DataFrame(rows)
            .sort_values(
                ["comments_count", "pain_units_count", "avg_confidence"],
                ascending=[False, False, False],
            )
            .reset_index(drop=True)
        )

    def build_comment_summary(
        self,
        units: pd.DataFrame,
        comments: pd.DataFrame,
    ) -> pd.DataFrame:
        if units.empty:
            result = comments.copy()
            result["has_extracted_pain"] = False
            return result

        rows: list[dict[str, Any]] = []
        for comment_id, part in units.groupby("comment_row_id"):
            pain_ids: list[str] = []
            for value in part["pain_ids"].fillna(""):
                for pain_id in self._split_labels(value):
                    if pain_id not in pain_ids:
                        pain_ids.append(pain_id)

            strongest = part.sort_values(
                "primary_combined_score",
                ascending=False,
            ).iloc[0]
            rows.append(
                {
                    "comment_row_id": comment_id,
                    "has_extracted_pain": True,
                    "extracted_pain_units_count": len(part),
                    "comment_pain_ids": "; ".join(pain_ids),
                    "comment_pain_titles": "; ".join(
                        self._catalog.get(pain_id).title for pain_id in pain_ids
                    ),
                    "strongest_pain_title": strongest["primary_pain_title"],
                    "strongest_pain_text": strongest["pain_text"],
                    "strongest_pain_score": strongest["primary_combined_score"],
                }
            )

        result = comments.merge(pd.DataFrame(rows), on="comment_row_id", how="left")
        result["has_extracted_pain"] = result["has_extracted_pain"].fillna(False)
        result["extracted_pain_units_count"] = (
            result["extracted_pain_units_count"].fillna(0).astype(int)
        )
        for column in (
            "comment_pain_ids",
            "comment_pain_titles",
            "strongest_pain_title",
            "strongest_pain_text",
        ):
            result[column] = result[column].fillna("")
        return result

    @staticmethod
    def build_query_summary(units: pd.DataFrame) -> pd.DataFrame:
        if units.empty or "query" not in units.columns:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        for query, part in units.groupby("query"):
            top_pains = part["primary_pain_title"].value_counts().head(5)
            rows.append(
                {
                    "query": query,
                    "pain_units_count": len(part),
                    "comments_count": part["comment_row_id"].nunique(),
                    "top_pains": " | ".join(
                        f"{title}: {count}" for title, count in top_pains.items()
                    ),
                }
            )
        return (
            pd.DataFrame(rows)
            .sort_values("pain_units_count", ascending=False)
            .reset_index(drop=True)
        )

    def _add_lemmas(self, dataframe: pd.DataFrame, text_column: str) -> pd.DataFrame:
        result = dataframe.copy()
        lemma_rows: list[str] = []

        for text in result[text_column].fillna(""):
            words = TextTools.LEMMA_WORD_PATTERN.findall(
                TextTools.normalize_for_rules(text)
            )
            tokens: list[str] = []
            for word in words:
                lemma = self._lemmatizer(word)
                if lemma not in STOPWORDS and len(lemma) >= 3:
                    tokens.append(lemma)
            lemma_rows.append(" ".join(tokens))

        result["lemmas"] = lemma_rows
        return result

    def _build_pain_summary_row(
        self,
        definition: PainDefinition,
        part: pd.DataFrame,
        *,
        total_comments: int,
    ) -> dict[str, Any]:
        base = {
            "pain_id": definition.pain_id,
            "pain_title": definition.title,
            "pain_question": definition.question,
            "pain_formula": definition.formula,
        }
        if part.empty:
            return {
                **base,
                "pain_units_count": 0,
                "comments_count": 0,
                "comments_share": 0.0,
                "avg_confidence": 0.0,
                "top_words": "",
                "top_phrases": "",
                "evidence_1": "",
                "evidence_2": "",
                "evidence_3": "",
            }

        comments_count = part["comment_row_id"].nunique()
        examples = (
            part.sort_values(
                ["primary_combined_score", "pain_likelihood_score"],
                ascending=[False, False],
            )
            .drop_duplicates(subset=["comment_row_id"])
            .head(3)["pain_text"]
            .tolist()
        )
        examples.extend([""] * (3 - len(examples)))

        return {
            **base,
            "pain_units_count": len(part),
            "comments_count": comments_count,
            "comments_share": round(comments_count / max(1, total_comments), 4),
            "avg_confidence": round(
                float(part["primary_combined_score"].mean()), 4
            ),
            "top_words": self._top_words(part),
            "top_phrases": self._top_phrases(part),
            "evidence_1": examples[0],
            "evidence_2": examples[1],
            "evidence_3": examples[2],
        }

    @staticmethod
    def _document_frequency(items: Iterable[Iterable[str]]) -> Counter[str]:
        counter: Counter[str] = Counter()
        for sequence in items:
            counter.update(set(sequence))
        return counter

    def _top_words(self, part: pd.DataFrame, limit: int = 12) -> str:
        counter = self._document_frequency(
            TextTools.safe_string(value).split()
            for value in part["lemmas"].fillna("")
        )
        return ", ".join(word for word, _ in counter.most_common(limit))

    def _top_phrases(self, part: pd.DataFrame, limit: int = 10) -> str:
        documents: list[list[str]] = []
        for lemmas in part["lemmas"].fillna(""):
            tokens = lemmas.split()
            phrases = [
                " ".join(tokens[index : index + size])
                for size in (2, 3)
                for index in range(len(tokens) - size + 1)
            ]
            documents.append(phrases)

        counter = self._document_frequency(documents)
        return ", ".join(phrase for phrase, _ in counter.most_common(limit))

    @staticmethod
    def _split_labels(value: Any) -> list[str]:
        return [item.strip() for item in TextTools.safe_string(value).split(";") if item.strip()]


class MarkdownReportBuilder:
    """Формирует человекочитаемый Markdown-отчёт из готовых таблиц."""

    def build(
        self,
        *,
        comments: pd.DataFrame,
        units: pd.DataFrame,
        summary: pd.DataFrame,
        query_summary: pd.DataFrame,
        preparation_statistics: Mapping[str, int],
    ) -> str:
        lines = [
            "# Анализ реальных болей предпринимателей",
            "",
            "## 1. Методика",
            "",
            "Единица анализа — не весь комментарий и не безымянный кластер, "
            "а конкретный фрагмент текста, где выражена боль: проблема, страх, "
            "нехватка, непонимание, препятствие или вопрос.",
            "",
            "```text",
            "бизнес-контекст + сигнал проблемы/страха/нехватки/вопроса = pain unit",
            "```",
            "",
            "После извлечения pain units BERT/SBERT сопоставляет каждый фрагмент "
            "с формулировками болей. Поэтому результат — это не темы вроде "
            "«налоги» или «маркетплейсы», а конкретные проблемные формулировки.",
            "",
            "## 2. Объём данных",
            "",
            f"- Строк на входе: **{preparation_statistics.get('input_rows', len(comments))}**",
            f"- Комментариев после фильтрации и дедупликации: **{len(comments)}**",
            "- Комментариев, где извлечена хотя бы одна боль: "
            f"**{units['comment_row_id'].nunique() if not units.empty else 0}**",
            f"- Извлечённых pain units: **{len(units)}**",
        ]

        self._append_platforms(lines, comments)
        self._append_query_summary(lines, query_summary)
        self._append_main_pains(lines, summary)
        self._append_pain_details(lines, summary)
        lines.extend(
            [
                "",
                "## 5. Ограничения",
                "",
                "- Это гибрид правил и BERT-сходства, а не обученная supervised-модель.",
                "- Категории не взаимоисключающие: один фрагмент может относиться "
                "к деньгам, маркетплейсам и марже одновременно.",
                "- На финальном этапе нужно вручную проверить `pain_units.csv` и "
                "`pain_analysis_audit.xlsx`, особенно спорные категории.",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _append_platforms(lines: list[str], comments: pd.DataFrame) -> None:
        if "platform" not in comments.columns:
            return
        lines.extend(["", "### Площадки"])
        for platform, count in comments["platform"].value_counts().head(10).items():
            lines.append(f"- {platform}: {count}")

    @staticmethod
    def _append_query_summary(lines: list[str], query_summary: pd.DataFrame) -> None:
        if query_summary.empty:
            return
        lines.extend(["", "### Запросы, где найдено больше всего pain units"])
        for _, row in query_summary.head(10).iterrows():
            lines.append(
                f"- {row['query']}: {int(row['pain_units_count'])} pain units; "
                f"топ: {TextTools.truncate(row['top_pains'], 250)}"
            )

    @staticmethod
    def _append_main_pains(lines: list[str], summary: pd.DataFrame) -> None:
        lines.extend(
            [
                "",
                "## 3. Главные боли",
                "",
                "| Боль | Комментариев | Доля комментариев | Типовой вопрос |",
                "|---|---:|---:|---|",
            ]
        )
        for _, row in summary.head(15).iterrows():
            if int(row["comments_count"]) == 0:
                continue
            lines.append(
                f"| {row['pain_title']} | {int(row['comments_count'])} | "
                f"{float(row['comments_share']):.1%} | "
                f"{TextTools.truncate(row['pain_question'], 180)} |"
            )

    @staticmethod
    def _append_pain_details(lines: list[str], summary: pd.DataFrame) -> None:
        lines.extend(["", "## 4. Подробности по болям"])
        for _, row in summary.head(12).iterrows():
            if int(row["comments_count"]) == 0:
                continue
            lines.extend(
                [
                    "",
                    f"### {row['pain_title']}",
                    "",
                    f"- Pain units: **{int(row['pain_units_count'])}**",
                    f"- Комментариев: **{int(row['comments_count'])}**",
                    f"- Доля комментариев: **{float(row['comments_share']):.1%}**",
                    f"- Типовой вопрос: {row['pain_question']}",
                    f"- Формула боли: `{row['pain_formula']}`",
                    f"- Частые слова: {TextTools.truncate(row['top_words'], 260)}",
                    f"- Частые фразы: {TextTools.truncate(row['top_phrases'], 260)}",
                    "",
                    "Доказательные фрагменты:",
                ]
            )
            for column in ("evidence_1", "evidence_2", "evidence_3"):
                if TextTools.safe_string(row[column]).strip():
                    lines.extend(["", f"> {TextTools.truncate(row[column])}"])


class FileResultsExporter:
    """Записывает CSV, Markdown и Excel-аудит на диск."""

    def __init__(self, catalog: PainCatalog) -> None:
        self._catalog = catalog

    def export(self, artifacts: AnalysisArtifacts, output_dir: Path) -> OutputPaths:
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = OutputPaths.inside(output_dir)

        artifacts.comments_with_pains.to_csv(
            paths.comments,
            index=False,
            encoding="utf-8-sig",
        )
        artifacts.pain_units.to_csv(
            paths.pain_units,
            index=False,
            encoding="utf-8-sig",
        )
        artifacts.rejected_fragments.to_csv(
            paths.rejected_fragments,
            index=False,
            encoding="utf-8-sig",
        )
        artifacts.pain_summary.to_csv(
            paths.pain_summary,
            index=False,
            encoding="utf-8-sig",
        )
        artifacts.query_summary.to_csv(
            paths.query_summary,
            index=False,
            encoding="utf-8-sig",
        )
        paths.report.write_text(artifacts.report, encoding="utf-8")
        self._write_audit_workbook(artifacts, paths.audit)
        return paths

    def _write_audit_workbook(
        self,
        artifacts: AnalysisArtifacts,
        path: Path,
    ) -> None:
        used_sheet_names: set[str] = set()
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            artifacts.pain_summary.to_excel(
                writer,
                sheet_name="pain_summary",
                index=False,
            )
            artifacts.query_summary.to_excel(
                writer,
                sheet_name="query_summary",
                index=False,
            )
            artifacts.pain_units.head(1000).to_excel(
                writer,
                sheet_name="pain_units_sample",
                index=False,
            )
            artifacts.rejected_fragments.head(1000).to_excel(
                writer,
                sheet_name="rejected_sample",
                index=False,
            )
            used_sheet_names.update(
                {
                    "pain_summary",
                    "query_summary",
                    "pain_units_sample",
                    "rejected_sample",
                }
            )

            for pain_id in self._catalog.ids():
                sample = artifacts.pain_units[
                    artifacts.pain_units["pain_ids"]
                    .fillna("")
                    .str.contains(pain_id, regex=False)
                ].head(100)
                if sample.empty:
                    continue
                sheet_name = self._unique_sheet_name(pain_id, used_sheet_names)
                sample.to_excel(writer, sheet_name=sheet_name, index=False)

    @staticmethod
    def _unique_sheet_name(name: str, used_names: set[str]) -> str:
        base = name[:31]
        candidate = base
        counter = 2
        while candidate in used_names:
            suffix = f"_{counter}"
            candidate = f"{base[:31 - len(suffix)]}{suffix}"
            counter += 1
        used_names.add(candidate)
        return candidate


class PainAnalysisPipeline:
    """Координатор сценария; вся предметная работа делегирована компонентам."""

    def __init__(
        self,
        *,
        table_reader: TableReader,
        preprocessor: CommentPreprocessor,
        extractor: PainUnitExtractor,
        encoder_factory: TextEncoderFactory,
        classifier: HybridPainClassifier,
        summary_builder: SummaryBuilder,
        report_builder: MarkdownReportBuilder,
        exporter: ResultsExporter,
    ) -> None:
        self._table_reader = table_reader
        self._preprocessor = preprocessor
        self._extractor = extractor
        self._encoder_factory = encoder_factory
        self._classifier = classifier
        self._summary_builder = summary_builder
        self._report_builder = report_builder
        self._exporter = exporter

    def run(self, config: AnalysisConfig) -> tuple[AnalysisArtifacts, OutputPaths]:
        config.validate()

        LOGGER.info("Загружаю таблицу: %s", config.input_path)
        raw_comments = self._table_reader.read(config.input_path)

        LOGGER.info("Фильтрую и дедуплицирую комментарии")
        prepared = self._preprocessor.prepare(
            raw_comments,
            use_existing_filter=config.use_existing_filter,
        )
        LOGGER.info("Комментариев после фильтрации: %s", len(prepared.dataframe))

        LOGGER.info("Извлекаю смысловые фрагменты с болями")
        extracted = self._extractor.extract(prepared.dataframe)
        LOGGER.info("Принято фрагментов: %s", len(extracted.accepted))
        LOGGER.info("Отклонено фрагментов: %s", len(extracted.rejected))
        if extracted.accepted.empty:
            raise ValueError(
                "Не удалось извлечь pain units. Ослабьте фильтры или проверьте "
                "содержимое входного файла."
            )

        encoder = self._encoder_factory.create(config.model_name)
        classified_units = self._classifier.classify(
            extracted.accepted,
            encoder,
            semantic_threshold=config.semantic_threshold,
            combined_threshold=config.combined_threshold,
            max_labels=config.max_labels,
        )

        LOGGER.info("Строю сводные таблицы")
        comments_with_pains = self._summary_builder.build_comment_summary(
            classified_units,
            prepared.dataframe,
        )
        pain_summary = self._summary_builder.build_pain_summary(
            classified_units,
            total_comments=len(prepared.dataframe),
        )
        query_summary = self._summary_builder.build_query_summary(classified_units)
        report = self._report_builder.build(
            comments=prepared.dataframe,
            units=classified_units,
            summary=pain_summary,
            query_summary=query_summary,
            preparation_statistics=prepared.statistics,
        )

        artifacts = AnalysisArtifacts(
            comments=prepared.dataframe,
            pain_units=classified_units,
            rejected_fragments=extracted.rejected,
            comments_with_pains=comments_with_pains,
            pain_summary=pain_summary,
            query_summary=query_summary,
            report=report,
            preparation_statistics=prepared.statistics,
        )
        paths = self._exporter.export(artifacts, config.output_dir)
        return artifacts, paths


class ApplicationFactory:
    """Единственная точка сборки зависимостей приложения."""

    @staticmethod
    def create(text_column: str = DEFAULT_TEXT_COLUMN) -> PainAnalysisPipeline:
        catalog = PainCatalog(PAIN_DEFINITIONS)
        preprocessor = CommentPreprocessor(text_column=text_column)
        extractor = PainUnitExtractor(
            splitter=FragmentSplitter(),
            evaluator=PainLikelihoodEvaluator(),
        )
        summary_builder = SummaryBuilder(catalog, RussianLemmatizer())
        return PainAnalysisPipeline(
            table_reader=PandasTableReader(),
            preprocessor=preprocessor,
            extractor=extractor,
            encoder_factory=SentenceTransformerEncoderFactory(),
            classifier=HybridPainClassifier(catalog),
            summary_builder=summary_builder,
            report_builder=MarkdownReportBuilder(),
            exporter=FileResultsExporter(catalog),
        )


class CliParser:
    """Преобразует аргументы командной строки в типизированную конфигурацию."""

    @staticmethod
    def parse(argv: Sequence[str] | None = None) -> AnalysisConfig:
        parser = argparse.ArgumentParser(
            description=(
                "Извлечение и гибридная BERT/SBERT-классификация "
                "бизнес-болей из комментариев."
            )
        )
        parser.add_argument("input_path", type=Path, help="Путь к CSV/XLSX")
        parser.add_argument(
            "--output-dir",
            type=Path,
            default=Path("bert_pain_results_v3"),
            help="Папка с результатами",
        )
        parser.add_argument(
            "--model",
            default=DEFAULT_MODEL,
            help="Имя SentenceTransformer-модели или псевдоним BERT/SBERT",
        )
        parser.add_argument("--semantic-threshold", type=float, default=0.36)
        parser.add_argument("--combined-threshold", type=float, default=0.34)
        parser.add_argument("--max-labels", type=int, default=3)
        parser.add_argument(
            "--no-existing-filter",
            action="store_true",
            help=(
                "Не использовать колонки suitable_for_pain_analysis "
                "и comment_type"
            ),
        )
        parser.add_argument(
            "--text-column",
            default=DEFAULT_TEXT_COLUMN,
            help="Название колонки с текстом комментария",
        )
        arguments = parser.parse_args(argv)

        return AnalysisConfig(
            input_path=arguments.input_path,
            output_dir=arguments.output_dir,
            model_name=MODEL_ALIASES.get(arguments.model, arguments.model),
            semantic_threshold=arguments.semantic_threshold,
            combined_threshold=arguments.combined_threshold,
            max_labels=arguments.max_labels,
            use_existing_filter=not arguments.no_existing_filter,
            text_column=arguments.text_column,
        )


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    config = CliParser.parse(argv)
    pipeline = ApplicationFactory.create(text_column=config.text_column)
    _, paths = pipeline.run(config)

    LOGGER.info("Готово. Основные файлы:")
    for path in (
        paths.report,
        paths.pain_summary,
        paths.pain_units,
        paths.comments,
        paths.audit,
    ):
        LOGGER.info("- %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
