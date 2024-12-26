from typing import Final, Dict
from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup, LabeledPrice, PreCheckoutQuery
)
from telegram.ext import (
    CommandHandler, CallbackQueryHandler, CallbackContext, Application,
    ContextTypes, MessageHandler, filters
)
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
PAYMENT_PROVIDER_TOKEN: Final = "YOUR_PAYMENT_PROVIDER_TOKEN"

# Store active predictions, user states, and subscription statuses
active_predictions: Dict[int, Dict] = {}
user_choices: Dict[int, Dict] = {}
user_subscription_status: Dict[int, bool] = {}

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

prediction_manager = PredictionManager()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Start command with subscription check."""
    user_id = update.effective_user.id

    # Check subscription status
    if user_id not in user_subscription_status:
        user_subscription_status[user_id] = False  # Default to not subscribed

    if user_subscription_status[user_id]:
        await send_crypto_options(update)
    else:
        await update.message.reply_text(
            "🚀 Welcome to Crypto Signal Bot!\n\n"
            "You can make one free prediction. After that, a subscription of 250 Telegram stars "
            "is required to use the bot.\n\n"
            "Type /subscribe to unlock full access."
        )

async def send_crypto_options(update: Update) -> None:
    """Display cryptocurrency options."""
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
        "Please select a cryptocurrency to begin:",
        reply_markup=reply_markup
    )

async def subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Initiate subscription process."""
    user_id = update.effective_user.id

    if user_subscription_status.get(user_id, False):
        await update.message.reply_text("✅ You are already subscribed!")
        return

    title = "Crypto Signal Bot Subscription"
    description = "Get unlimited access to cryptocurrency predictions."
    payload = f"subscription_{user_id}"
    currency = "USD"
    price = 250 * 100  # Telegram stars converted to minor units (e.g., cents)

    prices = [LabeledPrice("Subscription", price)]

    await context.bot.send_invoice(
        chat_id=update.effective_chat.id,
        title=title,
        description=description,
        payload=payload,
        provider_token=PAYMENT_PROVIDER_TOKEN,
        currency=currency,
        prices=prices,
        start_parameter="crypto_subscription"
    )

async def precheckout_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle pre-checkout queries."""
    query = update.pre_checkout_query
    await query.answer(ok=True)

async def successful_payment(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle successful payments."""
    user_id = update.effective_user.id
    user_subscription_status[user_id] = True
    await update.message.reply_text(
        "🎉 Thank you for subscribing! You now have unlimited access to predictions.\n\n"
        "Type /start to begin."
    )

async def crypto_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle cryptocurrency selection."""
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id

    if not user_subscription_status.get(user_id, False) and user_id in active_predictions:
        await query.edit_message_text(
            "❌ You have already used your free prediction.\n\n"
            "Type /subscribe to unlock unlimited access."
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

    crypto = user_choices[user_id]["crypto"]
    minutes_ahead = int(query.data)
    user_choices[user_id]["time"] = minutes_ahead

    await query.edit_message_text(
        f"🔄 Analyzing {crypto} data...\n"
        "This may take a moment while I train the prediction model."
    )

    try:
        # Run prediction asynchronously
        prediction, probabilities, current_price, current_time, confidence_level = (
            await prediction_manager.run_prediction(crypto, minutes_ahead)
        )

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
        logger.error(f"Prediction error: {str(e)}")
        await query.edit_message_text("❌ An error occurred. Please try again later.")

def main() -> None:
    """Initialize and start the bot."""
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("subscribe", subscribe))
    app.add_handler(CallbackQueryHandler(crypto_selection, pattern="^(BTC-USD|ETH-USD|SOL-USD|XRP-USD|DOGE-USD|DOT-USD)$"))
    app.add_handler(CallbackQueryHandler(time_selection, pattern="^(1|5|15)$"))
    app.add_handler(MessageHandler(filters.SUCCESSFUL_PAYMENT, successful_payment))
    app.add_handler(MessageHandler(filters.PRE_CHECKOUT_QUERY, precheckout_callback))

    logger.info("🚀 Bot is starting...")
    app.run_polling()

if __name__ == "__main__":
    main()
