import json
from typing import Final, Dict
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CommandHandler, CallbackQueryHandler, CallbackContext, Application, ContextTypes
import logging
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from formula import ShortTermPredictor

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TOKEN: Final = "7851729618:AAFs1mI8oaqql-3bekE4BJEHxE8AJ6_F2-c"
BOT_USERNAME: Final = "@signal_trading_test_bot"

# Store active predictions and user states
active_predictions: Dict[int, Dict] = {}
user_choices: Dict[int, Dict] = {}
user_data: Dict[int, Dict] = {}



class PredictionManager:
    def __init__(self):
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

    async def run_prediction(self, crypto: str, minutes: int) -> tuple:
        """Run prediction in a separate thread to avoid blocking."""
        try:
            predictor = ShortTermPredictor(crypto, minutes)

            # Run the CPU-intensive operations in a thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.thread_pool, predictor.train_model)

            # Get prediction
            prediction_result = await loop.run_in_executor(
                self.thread_pool,
                predictor.predict_next_movement
            )

            return prediction_result
        except Exception as e:
            logger.error(f"Prediction error for {crypto}: {str(e)}")
            raise

class UserManager:
    def __init__(self):
        self.data_file = "user_data.json"
        self.load_data()

    def load_data(self):
        try:
            with open(self.data_file, 'r') as f:
                global user_data
                user_data = json.load(f)
        except FileNotFoundError:
            user_data = {}

    def save_data(self):
        with open(self.data_file, 'w') as f:
            json.dump(user_data, f)

    def check_subscription(self, user_id: int) -> bool:
        user_info = user_data.get(str(user_id), {})
        return user_info.get('subscribed', False)

    def has_free_trial(self, user_id: int) -> bool:
        user_info = user_data.get(str(user_id), {})
        return not user_info.get('used_trial', False)

    def use_free_trial(self, user_id: int):
        if str(user_id) not in user_data:
            user_data[str(user_id)] = {}
        user_data[str(user_id)]['used_trial'] = True
        user_data[str(user_id)]['subscribed'] = False
        self.save_data()

    def subscribe_user(self, user_id: int):
        if str(user_id) not in user_data:
            user_data[str(user_id)] = {}
        user_data[str(user_id)]['subscribed'] = True
        self.save_data()

user_manager = UserManager()
prediction_manager = PredictionManager()

prediction_manager = PredictionManager()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id

    if not user_manager.check_subscription(user_id) and not user_manager.has_free_trial(user_id):
        subscription_message = (
            "🌟 Welcome to Crypto Signal Bot!\n\n"
            "To use this bot, you need a subscription (250 Telegram Stars).\n"
            "Please contact @admin to subscribe.\n\n"
            "New users get one free prediction - would you like to use it now?"
        )
        keyboard = [[
            InlineKeyboardButton("✨ Use Free Trial", callback_data="use_trial"),
            InlineKeyboardButton("💫 Subscribe", callback_data="subscribe")
        ]]
        await update.message.reply_text(
            subscription_message,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return

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
    welcome_message = "🚀 Select a cryptocurrency to begin:"
    await update.message.reply_text(welcome_message, reply_markup=reply_markup)


async def handle_subscription(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id

    if query.data == "use_trial":
        user_manager.use_free_trial(user_id)
        keyboard = [
            [
                InlineKeyboardButton("₿ BTC-USD", callback_data="BTC-USD"),
                InlineKeyboardButton("Ξ ETH-USD", callback_data="ETH-USD")
            ],
            [
                InlineKeyboardButton("◎ SOL-USD", callback_data="SOL-USD"),
                InlineKeyboardButton("✧ XRP-USD", callback_data="XRP-USD")
            ],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            "✨ Free trial activated! Select a cryptocurrency:",
            reply_markup=reply_markup
        )
    elif query.data == "subscribe":
        await query.edit_message_text(
            "💫 To subscribe:\n"
            "1. You need 250 Telegram Stars\n"
            "2. Contact @admin\n"
            "3. Once confirmed, you'll get unlimited predictions!\n\n"
            "Use /start to begin after subscribing."
        )

async def crypto_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id

    if not user_manager.check_subscription(user_id) and not user_manager.has_free_trial(user_id):
        await query.edit_message_text(
            "⭐ You need a subscription to continue.\n"
            "Contact @admin to subscribe (250 Telegram Stars required).\n\n"
            "Use /start to begin after subscribing."
        )
        return

    user_choices[user_id] = {"crypto": query.data}
    keyboard = [
        [
            InlineKeyboardButton("⚡ 1 Minute", callback_data="1"),
        ],
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
    """Handle time range selection and manage prediction process."""
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id

    if user_id not in user_choices:
        await query.edit_message_text("Session expired. Please start over with /start")
        return

    user_choices[user_id]["time"] = int(query.data)
    crypto = user_choices[user_id]["crypto"]
    minutes_ahead = user_choices[user_id]["time"]

    # Show loading message
    await query.edit_message_text(
        f"🔄 Analyzing {crypto} data...\n"
        "This may take a moment while I train the prediction model."
    )

    try:
        # Run prediction asynchronously
        prediction, probabilities, current_price, current_time, confidence_level = (
            await prediction_manager.run_prediction(crypto, minutes_ahead)
        )

        # Format prediction result
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
            "Type /start to make a new prediction"
        )

        # Store prediction in active_predictions
        active_predictions[user_id] = {
            "crypto": crypto,
            "timestamp": current_time,
            "prediction": prediction,
            "price": current_price
        }

        await query.edit_message_text(result_message)

    except Exception as e:
        error_message = (
            "❌ Sorry, an error occurred while making the prediction.\n"
            "This might be due to:\n"
            "• Market data unavailability\n"               
            "• Network connectivity issues\n"   
            "• Rate limiting\n\n"
            "Please try again later with /start"
        )
        logger.error(f"Prediction error for user {user_id}: {str(e)}")
        await query.edit_message_text(error_message)


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle errors gracefully."""
    logger.error(f"Update {update} caused error: {context.error}")
    error_message = (
        "❌ An unexpected error occurred.\n"
        "Please try again with /start"
    )

    try:
        if update and update.effective_message:
            await update.effective_message.reply_text(error_message)
        elif update and update.callback_query:
            await update.callback_query.edit_message_text(error_message)
    except Exception as e:
        logger.error(f"Error handler failed: {str(e)}")


def main() -> None:
    try:
        app = Application.builder().token(TOKEN).build()

        app.add_handler(CommandHandler("start", start))
        app.add_handler(CallbackQueryHandler(
            handle_subscription,
            pattern="^(use_trial|subscribe)$"
        ))
        app.add_handler(CallbackQueryHandler(
            crypto_selection,
            pattern="^(BTC-USD|ETH-USD|SOL-USD|XRP-USD|DOGE-USD|DOT-USD)$"
        ))
        app.add_handler(CallbackQueryHandler(
            time_selection,
            pattern="^(1|5|15)$"
        ))

        app.add_error_handler(error_handler)

        logger.info("🚀 Bot is starting...")
        app.run_polling()

    except Exception as e:
        logger.error(f"Failed to start bot: {str(e)}")
        raise


if __name__ == "__main__":
    main()