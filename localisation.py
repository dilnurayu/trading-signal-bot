LOCALIZATION = {
    'en': {
        'welcome': "🚀 Welcome to Crypto Signal Bot!\n\nI can help you predict short-term cryptocurrency price movements using machine learning.\n\nPlease select a cryptocurrency to begin:",
        'select_timeframe': "Selected: {}\nChoose prediction timeframe:",
        'analyzing': "🔄 Analyzing {} data...\nThis may take a moment while I train the prediction model.",
        'no_free_predictions': "You have no free predictions remaining.\nSubscribe for 250 Stars to continue using the bot.",
        'subscribe_button': "⭐ Subscribe - 250 Stars",
        'session_expired': "Session expired. Please start over with /start",
        'prediction_error': "❌ An error occurred while generating the prediction. Please try again later.",
        'subscription_thanks': "⭐ Thank you for subscribing! You now have full access for 30 days.",
        'remaining_predictions': "🎁 You have {} free predictions remaining.",
        'last_prediction': "❗ This was your last free prediction. Subscribe to continue using the bot.",
        'make_new_prediction': "\nType /crypto to make a new prediction",
        'timeframes': {
            1: "Short-term scalping prediction, expect high volatility",
            5: "Medium-term swing prediction, balanced between noise and trend",
            15: "Longer-term position prediction, better for trending markets"
        },
        'confidence_levels': {
            'High': {
                'recommendation': "Strong signal detected! This prediction has high confidence based on clear market patterns. Consider following the indicated direction while maintaining proper risk management."
            },
            'Medium': {
                'recommendation': "Moderate signal strength. While the direction shows promise, consider using smaller position sizes and tighter stop-losses. Monitor the market for additional confirmation signals."
            },
            'Low': {
                'recommendation': "Weak signal detected. The market shows mixed patterns. Consider waiting for a stronger signal or reducing your exposure. If trading, use minimal position sizes and strict risk management."
            },
            'Very Low': {
                'recommendation': "Market conditions are highly uncertain. The prediction confidence is very low, suggesting sideways or choppy market conditions. It's recommended to wait for clearer signals before taking any position."
            }
        },
        'prediction_result': (
            "🎯 Prediction for {crypto}\n\n"
            "⏰ Time: {time}\n"
            "💰 Current Price: ${price:,.6f}\n"
            "📈 Direction: {direction}\n"
            "🎲 Probability: {probability:.1f}%\n"
            "{emoji} Confidence: {confidence}\n"
            "⏱ Timeframe: Next {minutes} minute(s)\n"
            "ℹ️ {timeframe_context}\n\n"
            "📊 Recommendation:\n{recommendation}\n\n"
        )
    },
    'ru': {
        'welcome': "🚀 Добро пожаловать в Crypto Signal Bot!\n\nЯ помогу вам предсказать краткосрочные движения цен криптовалют с помощью машинного обучения.\n\nВыберите криптовалюту для начала:",
        'select_timeframe': "Выбрано: {}\nВыберите временной интервал:",
        'analyzing': "🔄 Анализ данных {}...\nЭто может занять некоторое время, пока я тренирую модель прогнозирования.",
        'no_free_predictions': "У вас не осталось бесплатных прогнозов.\nПодпишитесь за 250 Stars, чтобы продолжить использование бота.",
        'subscribe_button': "⭐ Подписаться - 250 Stars",
        'session_expired': "Сессия истекла. Начните заново с команды /start",
        'prediction_error': "❌ Произошла ошибка при генерации прогноза. Пожалуйста, попробуйте позже.",
        'subscription_thanks': "⭐ Спасибо за подписку! У вас теперь есть полный доступ на 30 дней.",
        'remaining_predictions': "🎁 У вас осталось {} бесплатных прогнозов.",
        'last_prediction': "❗ Это был ваш последний бесплатный прогноз. Подпишитесь, чтобы продолжить использование бота.",
        'make_new_prediction': "\nВведите /crypto для нового прогноза",
        'timeframes': {
            1: "Краткосрочный скальпинг прогноз, ожидается высокая волатильность",
            5: "Среднесрочный свинг прогноз, баланс между шумом и трендом",
            15: "Долгосрочный позиционный прогноз, лучше для трендовых рынков"
        },
        'confidence_levels': {
            'High': {
                'recommendation': "Обнаружен сильный сигнал! Этот прогноз имеет высокую уверенность на основе четких рыночных паттернов. Рекомендуется следовать указанному направлению с соблюдением правил управления рисками."
            },
            'Medium': {
                'recommendation': "Умеренная сила сигнала. Хотя направление выглядит многообещающим, рекомендуется использовать меньшие размеры позиций и более строгие стоп-лоссы. Следите за дополнительными сигналами подтверждения."
            },
            'Low': {
                'recommendation': "Обнаружен слабый сигнал. Рынок показывает смешанные паттерны. Рекомендуется дождаться более сильного сигнала или уменьшить экспозицию. При торговле используйте минимальные размеры позиций."
            },
            'Very Low': {
                'recommendation': "Рыночные условия крайне неопределенны. Уверенность прогноза очень низкая, что указывает на боковое или неустойчивое движение рынка. Рекомендуется дождаться более четких сигналов."
            }
        },
        'prediction_result': (
            "🎯 Прогноз для {crypto}\n\n"
            "⏰ Время: {time}\n"
            "💰 Текущая цена: ${price:,.2f}\n"
            "📈 Направление: {direction}\n"
            "🎲 Вероятность: {probability:.1f}%\n"
            "{emoji} Уверенность: {confidence}\n"
            "⏱ Временной интервал: Следующие {minutes} минут(ы)\n"
            "ℹ️ {timeframe_context}\n\n"
            "📊 Рекомендация:\n{recommendation}\n\n"
        )
    },
    'uz': {
        'welcome': "🚀 Crypto Signal Bot-ga xush kelibsiz!\n\nMen machine learning yordamida kriptovalyuta narxlarining qisqa muddatli harakatlarini bashorat qilishga yordam beraman.\n\nBoshlash uchun kriptovalyutani tanlang:",
        'select_timeframe': "Tanlangan: {}\nVaqt oralig'ini tanlang:",
        'analyzing': "🔄 {} ma'lumotlarini tahlil qilish...\nBashorat modelini o'rgatish uchun biroz vaqt kerak bo'ladi.",
        'no_free_predictions': "Bepul bashoratlar qolmadi.\nBotdan foydalanishni davom ettirish uchun 250 Stars evaziga obuna bo'ling.",
        'subscribe_button': "⭐ Obuna bo'lish - 250 Stars",
        'session_expired': "Sessiya muddati tugadi. /start buyrug'i bilan qayta boshlang",
        'prediction_error': "❌ Bashorat yaratishda xatolik yuz berdi. Iltimos, keyinroq qayta urinib ko'ring.",
        'subscription_thanks': "⭐ Obuna bo'lganingiz uchun rahmat! Endi sizda 30 kunlik to'liq kirish huquqi bor.",
        'remaining_predictions': "🎁 Sizda {} ta bepul bashorat qoldi.",
        'last_prediction': "❗ Bu sizning oxirgi bepul bashoratingiz edi. Botdan foydalanishni davom ettirish uchun obuna bo'ling.",
        'make_new_prediction': "\nYangi bashorat uchun /crypto ni bosing",
        'timeframes': {
            1: "Qisqa muddatli skalping bashorati, yuqori o'zgaruvchanlik kutilmoqda",
            5: "O'rta muddatli sving bashorati, shovqin va trend o'rtasidagi muvozanat",
            15: "Uzoq muddatli pozitsion bashorat, trend bozorlar uchun yaxshiroq"
        },
        'confidence_levels': {
            'High': {
                'recommendation': "Kuchli signal aniqlandi! Ushbu bashorat aniq bozor naqshlari asosida yuqori ishonchga ega. Ko'rsatilgan yo'nalishni risk boshqaruvi qoidalariga rioya qilgan holda kuzatib borish tavsiya etiladi."
            },
            'Medium': {
                'recommendation': "O'rtacha signal kuchi. Yo'nalish istiqbolli ko'rinsa ham, kichikroq pozitsiya hajmlari va qattiqroq stop-loss'lardan foydalanish tavsiya etiladi. Qo'shimcha tasdiqlash signallarini kuzatib boring."
            },
            'Low': {
                'recommendation': "Kuchsiz signal aniqlandi. Bozor aralash naqshlarni ko'rsatmoqda. Kuchliroq signalni kutish yoki ta'sirni kamaytirish tavsiya etiladi. Savdo qilishda minimal pozitsiya hajmlaridan foydalaning."
            },
            'Very Low': {
                'recommendation': "Bozor sharoiti juda noaniq. Bashorat ishonchi juda past, bu yon yoki beqaror bozor harakatini ko'rsatadi. Aniqroq signallarni kutish tavsiya etiladi."
            }
        },
        'prediction_result': (
            "🎯 {crypto} uchun bashorat\n\n"
            "⏰ Vaqt: {time}\n"
            "💰 Joriy narx: ${price:,.2f}\n"
            "📈 Yo'nalish: {direction}\n"
            "🎲 Ehtimollik: {probability:.1f}%\n"
            "{emoji} Ishonch: {confidence}\n"
            "⏱ Vaqt oralig'i: Keyingi {minutes} daqiqa\n"
            "ℹ️ {timeframe_context}\n\n"
            "📊 Tavsiya:\n{recommendation}\n\n"
        )
    }
}

# Add translation mappings for directions and confidence levels
DIRECTION_TRANSLATIONS = {
    'en': {'UP': '⬆️ UP', 'DOWN': '⬇️ DOWN'},
    'ru': {'UP': '⬆️ ВВЕРХ', 'DOWN': '⬇️ ВНИЗ'},
    'uz': {'UP': '⬆️ YUQORIGA', 'DOWN': '⬇️ PASTGA'}
}

CONFIDENCE_TRANSLATIONS = {
    'en': {'High': 'High', 'Medium': 'Medium', 'Low': 'Low', 'Very Low': 'Very Low'},
    'ru': {'High': 'Высокая', 'Medium': 'Средняя', 'Low': 'Низкая', 'Very Low': 'Очень низкая'},
    'uz': {'High': 'Yuqori', 'Medium': "O'rta", 'Low': 'Past', 'Very Low': 'Juda past'}
}