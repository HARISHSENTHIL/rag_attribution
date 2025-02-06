import asyncio
import json
import os
from datetime import datetime, timezone

from telethon import TelegramClient, events

# Telegram API credentials
API_ID = "24173837"  # Replace with your API ID
API_HASH = "92efb973e98db34515d87fc2415eafdf"  # Replace with your API hash
SESSION_NAME = "user_session"  # Session name for storing user credentials

# List of public channels to fetch messages from
CHANNELS = [
    "@geminsiders",
    "@TG_cryptostasher",
]

# Directory to store messages
OUTPUT_DIR = "social_trading_messages"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize Telegram client
client = TelegramClient(SESSION_NAME, API_ID, API_HASH)

def save_message(channel_name, message):
    """
    Appends a single message dict to a JSON file named after the channel.
    """
    channel_path = os.path.join(OUTPUT_DIR, f"{channel_name}.json")
    if os.path.exists(channel_path):
        with open(channel_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    data.append(message)

    with open(channel_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# Fetch only messages from November 1 (2023) onward
async def fetch_old_messages(channel):
    """
    Fetches historical messages from November 1, 2023 onwards from the given channel
    and saves them to a JSON file. Skips older messages.
    """
    print(f"Fetching old messages from {channel} (from Nov 1 onward)...")
    # Define the date limit (November 1, 2024, in UTC)
    date_limit = datetime(2024, 11, 1, tzinfo=timezone.utc)

    # We use reverse=True to go from oldest to newest
    async for message in client.iter_messages(channel, reverse=True):
        # Skip if no date or if message is older than our limit
        if not message.date or message.date < date_limit:
            continue
        
        # If it's a valid message with text, save it
        if message.text:
            message_data = {
                "channel_name": channel.strip("@"),
                "message_id": message.id,
                "date": str(message.date),
                "text": message.text,
            }
            save_message(channel.strip("@"), message_data)

    print(f"Completed fetching old messages from {channel}.")

# Listen for new messages (future messages)
@client.on(events.NewMessage(chats=CHANNELS))
async def new_message_listener(event):
    """
    Whenever a new message is posted in one of the monitored channels,
    save it to the corresponding JSON file.
    """
    channel = await event.get_chat()
    channel_name = channel.username or channel.title
    message_data = {
        "channel_name": channel_name.strip("@"),
        "message_id": event.message.id,
        "date": str(event.message.date),
        "text": event.message.message,
    }
    print(f"New message from {channel_name}: {event.message.message}")
    save_message(channel_name.strip("@"), message_data)

# Main function
async def main():
    # Start client (log in)
    await client.start()
    print("Logged in as:", await client.get_me())

    # Fetch historical messages (from Nov 1 onward) for each channel
    for channel in CHANNELS:
        await fetch_old_messages(channel)

    # Keep listening for new messages
    print("Listening for new messages...")
    await client.run_until_disconnected()

# Run the script
if __name__ == "__main__":
    asyncio.run(main())
