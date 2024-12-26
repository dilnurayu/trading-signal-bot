from typing import Final, Dict
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, LabeledPrice, PreCheckoutQuery
from telegram.ext import (
    CommandHandler, CallbackQueryHandler, Application,
    ContextTypes, MessageHandler, filters, PreCheckoutQueryHandler
)
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import mysql.connector
from datetime import datetime, timedelta
from formula import ShortTermPredictor

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TOKEN: Final = "7464820680:AAHeYmWzf88-7KDs8BiJF8liG6sQDeN31Zc"
BOT_USERNAME: Final = "@signal_trading_ai_bot"
user_choices: Dict[int, Dict] = {}

# Database configuration
DB_CONFIG = {
    'host': 'develosh.beget.tech',
    'user': 'develosh_trading',
    'password': 'S2h0E0r4',
    'database': 'develosh_trading'
}


class DatabaseManager:
    def __init__(self):
        self.conn = None
        self.cursor = None
        self.connect()

    def connect(self):
        try:
            if self.conn is None or not self.conn.is_connected():
                self.conn = mysql.connector.connect(
                    host='develosh.beget.tech',
                    user='develosh_trading',
                    password='Shaha2001',
                    database='develosh_trading'
                )
                self.cursor = self.conn.cursor(dictionary=True)
                logger.info("Successfully connected to database")
        except mysql.connector.Error as err:
            logger.error(f"Database connection error: {err}")
            raise

    def ensure_connection(self):
        try:
            if self.conn is None or not self.conn.is_connected():
                self.connect()
        except mysql.connector.Error as err:
            logger.error(f"Database reconnection error: {err}")
            raise

    def get_user(self, user_id: int) -> dict:
        try:
            self.ensure_connection()
            self.cursor.execute(
                "SELECT * FROM users WHERE id = %s", (user_id,)
            )
            return self.cursor.fetchone()
        except mysql.connector.Error as err:
            logger.error(f"Database query error: {err}")
            raise

    def create_user(self, user_id: int) -> None:
        try:
            self.ensure_connection()
            self.cursor.execute(
                """INSERT INTO users (id, available_free_request, subscribe_type, created_at) 
                VALUES (%s, 3, 0, NOW())""", (user_id,)
            )
            self.conn.commit()
        except mysql.connector.Error as err:
            logger.error(f"Database insert error: {err}")
            self.conn.rollback()
            raise

    def decrease_available_request(self, user_id: int) -> None:
        try:
            self.ensure_connection()
            self.cursor.execute(
                "UPDATE users SET available_free_request = available_free_request - 1 WHERE id = %s",
                (user_id,)
            )
            self.conn.commit()
        except mysql.connector.Error as err:
            logger.error(f"Database update error: {err}")
            self.conn.rollback()
            raise

    def add_subscription(self, user_id: int, sub_type: int, duration_days: int) -> None:
        try:
            self.ensure_connection()
            now = datetime.now()
            end_date = now + timedelta(days=duration_days)
            self.cursor.execute(
                """UPDATE users SET 
                subscribe_type = %s,
                subscribed_at = %s,
                subscribe_end = %s 
                WHERE id = %s""",
                (sub_type, now, end_date, user_id)
            )
            self.conn.commit()
        except mysql.connector.Error as err:
            logger.error(f"Database subscription error: {err}")
            self.conn.rollback()
            raise

    def check_subscription_status(self, user_id: int) -> bool:
        try:
            self.ensure_connection()
            self.cursor.execute(
                """SELECT subscribe_type, subscribe_end 
                FROM users WHERE id = %s""",
                (user_id,)
            )
            result = self.cursor.fetchone()
            if not result or not result['subscribe_type']:
                return False
            return result['subscribe_end'] > datetime.now()
        except mysql.connector.Error as err:
            logger.error(f"Database subscription check error: {err}")
            return False

    def __del__(self):
        try:
            if hasattr(self, 'cursor') and self.cursor:
                self.cursor.close()
            if hasattr(self, 'conn') and self.conn:
                self.conn.close()
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")

db = DatabaseManager()

class PredictionManager:
    def __init__(self):
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

    async def run_prediction(self, crypto: str, minutes: int) -> tuple:
        try:
            predictor = ShortTermPredictor(crypto, minutes)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.thread_pool, predictor.train_model)
            prediction_result = await loop.run_in_executor(
                self.thread_pool,
                predictor.predict_next_movement
            )
            return prediction_result
        except Exception as e:
            logger.error(f"Prediction error for {crypto}: {str(e)}")
            raise

prediction_manager = PredictionManager()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id
    user = db.get_user(user_id)

    if not user:
        db.create_user(user_id)
        user = db.get_user(user_id)

    if user['available_free_request'] <= 0 and not db.check_subscription_status(user_id):
        keyboard = [
            [InlineKeyboardButton("⭐ Subscribe - 250 Stars", callback_data="subscribe")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(
            "You have no free predictions remaining.\n"
            "Subscribe for 250 Stars to continue using the bot.",
            reply_markup=reply_markup
        )
    else:
        await send_crypto_options(update)

async def send_crypto_options(update: Update) -> None:
    keyboard = [
        [
            InlineKeyboardButton("₿ BTC-USD", callback_data="BTC-USD"),
            InlineKeyboardButton("Ξ ETH-USD", callback_data="ETH-USD")
        ],
        [
            InlineKeyboardButton("◎ SOL-USD", callback_data="SOL-USD"),
            InlineKeyboardButton("✧ XRP-USD", callback_data="XRP-USD")
        ],
        [
            InlineKeyboardButton("◎ DOGE-USD", callback_data="DOGE-USD"),
            InlineKeyboardButton("✧ DOT-USD", callback_data="DOT-USD")
        ],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "🚀 Welcome to Crypto Signal Bot!\n\nI can help you predict short-term cryptocurrency price movements using machine learning.\n\nPlease select a cryptocurrency to begin:",
        reply_markup=reply_markup
    )

async def handle_subscription(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    prices = [LabeledPrice("Bot Subscription", 250)]
    await context.bot.send_invoice(
        chat_id=query.from_user.id,
        title="Crypto Signal Bot Subscription",
        description="30 days of full access to predictions",
        payload="subscription_30_days",
        provider_token="YOUR_PAYMENT_PROVIDER_TOKEN",
        currency="XTR",
        prices=prices
    )


async def pre_checkout_query_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query: PreCheckoutQuery = update.pre_checkout_query
    await query.answer(ok=True)


async def successful_payment(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id
    db.add_subscription(user_id, 1, 30)  # Type 1 subscription for 30 days
    await update.message.reply_text(
        "⭐ Thank you for subscribing! You now have full access for 30 days."
    )


async def crypto_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    user = db.get_user(user_id)

    if not db.check_subscription_status(user_id) and user['available_free_request'] <= 0:
        await query.edit_message_text(
            "❌ You have no free predictions remaining.\n\n"
            "Subscribe to unlock unlimited access for 250 Stars."
        )
        return

    user_choices[user_id] = {"crypto": query.data}
    keyboard = [
        [InlineKeyboardButton("⚡ 1 Minute", callback_data="1")],
        [
            InlineKeyboardButton("🕐 5 Minutes", callback_data="5"),
            InlineKeyboardButton("🕑 15 Minutes", callback_data="15")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(
        f"Selected: {query.data}\nChoose prediction timeframe:",
        reply_markup=reply_markup
    )


async def time_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    user = db.get_user(user_id)

    if not db.check_subscription_status(user_id) and user['available_free_request'] <= 0:
        await query.edit_message_text(
            "❌ You have no free predictions remaining.\n"
            "Subscribe to continue using the bot."
        )
        return

    if user_id not in user_choices:
        await query.edit_message_text("Session expired. Please start over with /start")
        return

    crypto = user_choices[user_id]["crypto"]
    minutes_ahead = int(query.data)
    user_choices[user_id]["time"] = minutes_ahead

    await query.edit_message_text(
        f"🔄 Analyzing {crypto} data...\n"
        "This may take a moment while I train the prediction model."
    )

    try:
        prediction, probabilities, current_price, current_time, confidence_level = (
            await prediction_manager.run_prediction(crypto, minutes_ahead)
        )

        if not db.check_subscription_status(user_id):
            db.decrease_available_request(user_id)
            user = db.get_user(user_id)

        direction = "⬆️ UP" if prediction == 1 else "⬇️ DOWN"
        probability = probabilities[1] * 100 if prediction == 1 else probabilities[0] * 100
        confidence_emoji = "🟢" if confidence_level == "High" else "🟡"

        result_message = (
            f"🎯 Prediction for {crypto}\n\n"
            f"⏰ Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"💰 Current Price: ${float(current_price):,.2f}\n"
            f"📈 Direction: {direction}\n"
            f"🎲 Probability: {probability:.1f}%\n"
            f"{confidence_emoji} Confidence: {confidence_level}\n"
            f"⏱ Timeframe: Next {minutes_ahead} minute(s)\n\n"
        )

        if not db.check_subscription_status(user_id):
            remaining = user['available_free_request']
            if remaining > 0:
                result_message += f"\n🎁 You have {remaining} free predictions remaining."
            else:
                result_message += "\n❗ This was your last free prediction. Subscribe to continue using the bot."

        result_message += "\nType /start to make a new prediction"

        await query.edit_message_text(result_message)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        await query.edit_message_text("❌ An error occurred. Please try again later.")


def main() -> None:
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(
        CallbackQueryHandler(crypto_selection, pattern="^(BTC-USD|ETH-USD|SOL-USD|XRP-USD|DOGE-USD|DOT-USD)$"))
    app.add_handler(CallbackQueryHandler(time_selection, pattern="^(1|5|15)$"))
    app.add_handler(CallbackQueryHandler(handle_subscription, pattern="^subscribe$"))
    app.add_handler(PreCheckoutQueryHandler(pre_checkout_query_handler))
    app.add_handler(MessageHandler(filters.SUCCESSFUL_PAYMENT, successful_payment))

    logger.info("🚀 Bot is starting...")
    app.run_polling()


if __name__ == "__main__":
    main()